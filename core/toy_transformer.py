from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

class ToyTransformer(nn.Module):

    def __init__(
        self,
        arch: str | type[PretrainedConfig] | PretrainedConfig = "qwen2",
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int | None = None,
        d_mlp: int | None = None,
        n_ctx: int = 128,
        tie_embeddings: bool = False,
        init_std: float = 0.02,
        attn_dropout: float = 0.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **config_kwargs,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.init_std = init_std
        self.config = self._build_config(
            arch,
            vocab_size=0,
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads or n_heads,
            intermediate_size=d_mlp if d_mlp is not None else 4 * d_model,
            max_position_embeddings=n_ctx,
            tie_word_embeddings=tie_embeddings,
            initializer_range=init_std,
            attention_dropout=attn_dropout,
            bos_token_id=None,
            eos_token_id=None,
            **config_kwargs,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Initializing zero-element")
            model = AutoModelForCausalLM.from_config(self.config)
        self.model = model.to(device=device, dtype=dtype)

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.tasks: dict[str, dict] = {}

    @staticmethod
    def _build_config(
        arch: str | type[PretrainedConfig] | PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        if isinstance(arch, PretrainedConfig):
            arch.vocab_size = 0
            return arch
        if isinstance(arch, str):
            if arch not in CONFIG_MAPPING:
                raise ValueError(
                    f"unknown architecture {arch!r}; expected an HF model_type such as "
                    "'qwen2', 'llama' or 'mistral'"
                )
            arch = CONFIG_MAPPING[arch]
        config = arch(**kwargs)
        for key, value in kwargs.items():
            if getattr(config, key, None) != value:
                warnings.warn(
                    f"{type(config).__name__} ignored or renamed {key}={value!r}; "
                    "pass the architecture's own field name via config_kwargs"
                )
        return config

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def n_ctx(self) -> int:
        return self.config.max_position_embeddings

    @property
    def device(self) -> torch.device:
        return self.model.device

    def tokenize(self, tokens: Sequence[str], batched: bool = True) -> torch.Tensor:
        try:
            ids = [self.token_to_id[t] for t in tokens]
        except KeyError as err:
            raise KeyError(f"token {err.args[0]!r} is not in the vocabulary") from None
        out = torch.tensor(ids, dtype=torch.long, device=self.device)
        return out.unsqueeze(0) if batched else out

    def detokenize(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    def _grow_vocab(self, tokens: Sequence[str]) -> list[int]:
        fresh = [t for t in dict.fromkeys(tokens) if t not in self.token_to_id]
        if fresh:
            for tok in fresh:
                self.token_to_id[tok] = len(self.id_to_token)
                self.id_to_token.append(tok)
            self.model.resize_token_embeddings(self.vocab_size, mean_resizing=False)
            with torch.no_grad():
                matrices = [self.model.get_input_embeddings().weight]
                if not self.config.tie_word_embeddings:
                    matrices.append(self.model.get_output_embeddings().weight)
                for matrix in matrices:
                    nn.init.normal_(matrix[-len(fresh) :], std=self.init_std)
        return [self.token_to_id[t] for t in tokens]

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.vocab_size == 0:
            raise RuntimeError(
                "vocabulary is empty; register at least one task with add_task() first"
            )
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.shape[-1] > self.n_ctx:
            raise ValueError(
                f"sequence length {tokens.shape[-1]} exceeds n_ctx={self.n_ctx}"
            )
        return self.model(input_ids=tokens, **kwargs).logits

    def loss(self, tokens: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        labels = tokens.masked_fill(tokens == ignore_index, -100)
        return self.model(input_ids=tokens, labels=labels).loss

    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 10, temperature: float = 1.0
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        tokens = prompt if prompt.ndim == 2 else prompt.unsqueeze(0)
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.n_ctx :])[:, -1]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = (logits / temperature).softmax(dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=-1)
        self.train(was_training)
        return tokens

    def add_task(self, name: str, tokens: Sequence[str], **metadata) -> dict:
        if name in self.tasks:
            raise ValueError(f"task {name!r} is already registered")
        ids = self._grow_vocab(list(tokens))
        self.tasks[name] = {
            "name": name,
            "token_ids": dict(zip(tokens, ids)),
            **metadata,
        }
        return self.tasks[name]

    def train_task(self, name: str, **train_kwargs):
        if name not in self.tasks:
            raise KeyError(f"unknown task {name!r}; call add_task() first")
        raise NotImplementedError("train_task is not implemented yet")

    def __repr__(self) -> str:
        c = self.config
        return (
            f"{type(self).__name__}({c.model_type}, d_model={c.hidden_size}, "
            f"n_layers={c.num_hidden_layers}, n_heads={c.num_attention_heads}, "
            f"n_ctx={self.n_ctx}, vocab_size={self.vocab_size}, "
            f"n_tasks={len(self.tasks)})"
        )

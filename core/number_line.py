from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Iterable, Sequence

TRUE_TOKEN = "TRUE"
FALSE_TOKEN = "FALSE"


class NumberLine:

    TRUE_TOKEN = TRUE_TOKEN
    FALSE_TOKEN = FALSE_TOKEN

    def __init__(
        self,
        d: int,
        alpha: float = 1.0,
        item_names: Sequence[str] | None = None,
        rank_order: Sequence[int] | None = None,
        shuffle_ranks: bool = False,
        seed: int | None = None,
    ):
        if d < 1:
            raise ValueError(f"d must be at least 1, got {d}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.d = d
        self.alpha = alpha
        self.rng = random.Random(seed)

        if item_names is None:
            item_names = [str(i) for i in range(d)]
        elif len(item_names) != d:
            raise ValueError(f"expected {d} item names, got {len(item_names)}")
        if len(set(item_names)) != d:
            raise ValueError("item names must be unique")
        if TRUE_TOKEN in item_names or FALSE_TOKEN in item_names:
            raise ValueError(f"{TRUE_TOKEN!r} and {FALSE_TOKEN!r} are reserved")

        self.items = list(item_names)
        self.position = {name: i for i, name in enumerate(self.items)}

        if rank_order is None:
            rank_order = list(range(d))
            if shuffle_ranks:
                self.rng.shuffle(rank_order)
        elif sorted(rank_order) != list(range(d)):
            raise ValueError("rank_order must be a permutation of range(d)")
        self.rank_order = list(rank_order)

        self.item_weights = [0.0] * d
        for rank, pos in enumerate(self.rank_order):
            self.item_weights[pos] = (rank + 1) ** -alpha
        total = sum(self.item_weights)
        self.item_probs = [w / total for w in self.item_weights]

        self.operators: dict[str, dict] = {}
        self.token_counts: Counter[str] = Counter()

    @property
    def operator_tokens(self) -> list[str]:
        return list(self.operators)

    @property
    def vocab(self) -> list[str]:
        return self.items + self.operator_tokens + [TRUE_TOKEN, FALSE_TOKEN]

    def add_operator_token(
        self,
        name: str,
        fn: Callable[..., bool | int | str],
        arity: int = 2,
        weight: float = 1.0,
        **metadata,
    ) -> dict:
        if name in self.operators:
            raise ValueError(f"operator {name!r} is already registered")
        if name in self.position or name in (TRUE_TOKEN, FALSE_TOKEN):
            raise ValueError(f"operator name {name!r} collides with an existing token")
        if arity < 1:
            raise ValueError(f"arity must be at least 1, got {arity}")
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")
        self.operators[name] = {
            "name": name,
            "fn": fn,
            "arity": arity,
            "weight": weight,
            **metadata,
        }
        return self.operators[name]

    def item_frequency(self, token: str) -> float:
        return self.item_probs[self.position[token]]

    def sample_items(self, n: int = 1) -> list[str]:
        return self.rng.choices(self.items, weights=self.item_weights, k=n)

    def sample_operator(self) -> str:
        if not self.operators:
            raise RuntimeError("no operators registered; call add_operator_token() first")
        names = self.operator_tokens
        weights = [self.operators[n]["weight"] for n in names]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def evaluate(self, name: str, operands: Sequence[str]) -> str:
        op = self.operators[name]
        if len(operands) != op["arity"]:
            raise ValueError(
                f"operator {name!r} takes {op['arity']} operands, got {len(operands)}"
            )
        result = op["fn"](*(self.position[x] for x in operands))
        if isinstance(result, bool):
            return TRUE_TOKEN if result else FALSE_TOKEN
        if isinstance(result, int):
            if not 0 <= result < self.d:
                raise ValueError(
                    f"operator {name!r} returned position {result}, "
                    f"outside the number line [0, {self.d})"
                )
            return self.items[result]
        if result in self.position or result in (TRUE_TOKEN, FALSE_TOKEN):
            return result
        raise ValueError(f"operator {name!r} returned an unusable value {result!r}")

    def make_instance(self, name: str | None = None) -> list[str]:
        name = self.sample_operator() if name is None else name
        if name not in self.operators:
            raise KeyError(f"unknown operator {name!r}")
        operands = self.sample_items(self.operators[name]["arity"])
        sequence = [name, *operands, self.evaluate(name, operands)]
        self.token_counts.update(sequence)
        return sequence

    def training_instances(
        self, n: int, operators: Iterable[str] | None = None
    ) -> list[list[str]]:
        if operators is None:
            return [self.make_instance() for _ in range(n)]
        names = list(operators)
        return [self.make_instance(names[i % len(names)]) for i in range(n)]

    def enumerate_instances(self, operators: Iterable[str] | None = None) -> list[list[str]]:
        names = self.operator_tokens if operators is None else list(operators)
        out = []
        for name in names:
            arity = self.operators[name]["arity"]
            if arity != 2:
                raise NotImplementedError(
                    f"enumerate_instances supports arity-2 operators; {name!r} has {arity}"
                )
            for a in self.items:
                for b in self.items:
                    out.append([name, a, b, self.evaluate(name, [a, b])])
        return out

    def empirical_frequencies(self) -> dict[str, float]:
        total = sum(self.token_counts.values())
        if not total:
            return {token: 0.0 for token in self.vocab}
        return {token: self.token_counts[token] / total for token in self.vocab}

    def reset_counts(self) -> None:
        self.token_counts.clear()

    def export(self, n: int = 0) -> dict:
        return {
            "vocab": self.vocab,
            "instances": self.training_instances(n) if n else [],
            "item_probs": dict(zip(self.items, self.item_probs)),
        }

    def __len__(self) -> int:
        return self.d

    def __repr__(self) -> str:
        return (
            f"NumberLine(d={self.d}, alpha={self.alpha}, "
            f"n_operators={len(self.operators)}, vocab_size={len(self.vocab)})"
        )

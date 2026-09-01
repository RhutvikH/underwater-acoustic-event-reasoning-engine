"""Compact NSGA-II over PolicyVector. Seed-locked. No pymoo dependency."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from uaere.types import ObjectiveVector, PolicyVector


def nsga2(
    evaluate: Callable[[PolicyVector], ObjectiveVector],
    seed: int = 0,
    pop_size: int = 24,
    n_gen: int = 16,
    devices: tuple[str, ...] = ("stm32l476",),
) -> list[tuple[PolicyVector, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pop = [_random_policy(rng, devices) for _ in range(pop_size)]
    F = [evaluate(p).as_array() for p in pop]
    for _ in range(n_gen):
        child: list[PolicyVector] = []
        while len(child) < pop_size:
            a, b = rng.integers(0, pop_size, size=2)
            p1, p2 = pop[int(a)], pop[int(b)]
            child.append(_crossover_mutate(p1, p2, rng, devices))
        Fc = [evaluate(p).as_array() for p in child]
        merged_p = pop + child
        merged_f = F + Fc
        ranks = _non_dominated_ranks(np.stack(merged_f))
        order = np.lexsort((_crowding(np.stack(merged_f), ranks), ranks))
        pick = order[:pop_size]
        pop = [merged_p[i] for i in pick]
        F = [merged_f[i] for i in pick]
    ranks = _non_dominated_ranks(np.stack(F))
    front = [(pop[i], F[i]) for i in range(len(pop)) if ranks[i] == 0]
    return front or list(zip(pop, F, strict=True))


def _random_policy(rng: np.random.Generator, devices: tuple[str, ...]) -> PolicyVector:
    t1, t2, t3 = np.sort(rng.uniform(0.15, 0.9, size=3))
    return PolicyVector(
        tau1=float(t1),
        tau2=float(t2),
        tau3=float(t3),
        device_id=str(rng.choice(devices)),
        tau_collab_lo=float(rng.uniform(0.3, 0.45)),
        tau_collab_hi=float(rng.uniform(0.5, 0.65)),
        require_authenticated=True,
    )


def _crossover_mutate(
    a: PolicyVector, b: PolicyVector, rng: np.random.Generator, devices: tuple[str, ...]
) -> PolicyVector:
    mix = rng.random()
    t1 = mix * a.tau1 + (1 - mix) * b.tau1 + float(rng.normal(0, 0.03))
    t2 = mix * a.tau2 + (1 - mix) * b.tau2 + float(rng.normal(0, 0.03))
    t3 = mix * a.tau3 + (1 - mix) * b.tau3 + float(rng.normal(0, 0.03))
    t1, t2, t3 = np.sort(np.clip([t1, t2, t3], 0.05, 0.95))
    return PolicyVector(
        tau1=float(t1),
        tau2=float(t2),
        tau3=float(t3),
        device_id=str(rng.choice([a.device_id, b.device_id, rng.choice(devices)])),
        require_authenticated=True,
    )


def _non_dominated_ranks(F: np.ndarray) -> np.ndarray:
    n = len(F)
    ranks = np.zeros(n, dtype=int)
    remaining = set(range(n))
    r = 0
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            ranks[i] = r
            remaining.remove(i)
        r += 1
        if r > n:
            break
    return ranks


def _crowding(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    n, m = F.shape
    crowd = np.zeros(n)
    for r in np.unique(ranks):
        idx = np.where(ranks == r)[0]
        if len(idx) <= 2:
            crowd[idx] = np.inf
            continue
        block = F[idx]
        c = np.zeros(len(idx))
        for k in range(m):
            order = np.argsort(block[:, k])
            c[order[0]] = c[order[-1]] = np.inf
            span = block[order[-1], k] - block[order[0], k]
            if span <= 0:
                continue
            for i in range(1, len(idx) - 1):
                c[order[i]] += (block[order[i + 1], k] - block[order[i - 1], k]) / span
        crowd[idx] = c
    return -crowd  # lexsort wants smaller-is-better; we invert so large crowding first

"""Dung-style abstract argumentation over marine causes.

An attack fires unless the environment satisfies `unless_sea_state_ge`.
The grounded extension is computed by iterating: a cause is rejected if
an attack on it fires. WHY / WHY-NOT sentences are just the accepted set
and the fired attacks — no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from uaere.kg.graph import MarineAcousticKG
from uaere.types import EnvironmentState


@dataclass(frozen=True)
class Attack:
    attacker: str
    attacked: str
    unless_sea_state_ge: int | None = None


@dataclass
class ArgumentationResult:
    accepted: list[str]
    rejected: list[str]
    attacks_fired: list[str]
    why: str
    why_not: list[str]


def grounded_extension(
    candidates: list[str],
    attacks: list[Attack],
    sea_state: int,
) -> ArgumentationResult:
    cand = list(dict.fromkeys(candidates))
    fired: list[Attack] = []
    rejected: set[str] = set()
    for atk in attacks:
        need = atk.unless_sea_state_ge
        if need is not None and sea_state >= need:
            continue
        if atk.attacked in cand:
            fired.append(atk)
            rejected.add(atk.attacked)
    accepted = [c for c in cand if c not in rejected]
    why_not = [
        f"{a.attacked} attacked by {a.attacker} (sea-state {sea_state}"
        + (f" < {a.unless_sea_state_ge})" if a.unless_sea_state_ge is not None else ")")
        for a in fired
    ]
    why = "accepted: " + (", ".join(accepted) if accepted else "(none)")
    return ArgumentationResult(
        accepted=accepted,
        rejected=sorted(rejected),
        attacks_fired=[f"{a.attacker}->{a.attacked}" for a in fired],
        why=why,
        why_not=why_not,
    )


def attacks_from_kg(kg: MarineAcousticKG, sea_state: int) -> list[Attack]:
    out: list[Attack] = []
    for u, v, data in kg.g.edges(data=True):
        if data.get("rel") != "attacks":
            continue
        unless = data.get("unless_sea_state_ge")
        unless_i = int(unless) if unless is not None else None
        out.append(Attack(attacker=u, attacked=v, unless_sea_state_ge=unless_i))
    _ = sea_state
    return out


def argue_causes(
    kg: MarineAcousticKG,
    cause_ids: list[str],
    state: EnvironmentState,
    event_id: str | None = None,
) -> ArgumentationResult:
    atks = attacks_from_kg(kg, state.sea_state)
    if event_id is not None:
        allowed = set(cause_ids) | {event_id}
        atks = [a for a in atks if a.attacker in allowed]
    return grounded_extension(cause_ids, atks, state.sea_state)

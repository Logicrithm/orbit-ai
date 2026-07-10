"""
match.py — Orbit's matching funnel: who should this person meet, and why?

Three stages, each cheaper than the next is smarter:
  1. Hard filters (code)  — intent + dealbreakers. Never weighted, never overridden.
  2. Scoring (math)       — embeddings + overlap. 150 -> top 12 in milliseconds.
  3. Rerank (Gemma)       — reads the finalists, picks 3, names the connection
                            type, writes the reason and a first activity.

Design rule carried through: numbers never enter or leave the model.
Profiles are converted to WORDS before Gemma sees them (same reason extraction
outputs leanings, not values: a 4B model is reliable with words, not scales).
Output is never a percentage. A name, a type, a reason, an activity.
"""

import json

import numpy as np

from llm import chat_json

# ---------------------------------------------------------------------------
# Embeddings — local, CPU, no second model server. Loaded once, lazily.
# ---------------------------------------------------------------------------

_embedder = None


def embed(texts: list[str]) -> np.ndarray:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder.encode(texts, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Stage 1 — hard filters. Boolean. A "no" here cannot be bought back by score.
# ---------------------------------------------------------------------------

def _profile_text(p: dict) -> str:
    return " ".join([
        p.get("direction", ""),
        " ".join(p.get("values", [])),
        " ".join(p.get("strengths", [])),
        " ".join(p.get("gaps", [])),
        " ".join(p.get("dealbreakers", [])),
    ]).lower()


def passes_hard_filters(user: dict, cand: dict) -> bool:
    # intent: must want the same kind of connection (unsure is compatible with anything)
    ui, ci = user.get("intent"), cand.get("intent")
    if ui and ci and "unsure" not in (ui, ci) and ui != ci:
        return False

    # dealbreakers: stem-level keyword screen against the candidate's own text.
    # Stems, because "discrimination" must catch "discriminate", "discriminatory".
    # Honest limitation (README): catches only what candidates SAY about
    # themselves; stage 3 re-checks semantically.
    cand_words = _profile_text(cand).split()
    for db in user.get("dealbreakers", []):
        for word in db.lower().split():
            if len(word) <= 4:
                continue
            stem = word[:6]
            if any(cw.startswith(stem) for cw in cand_words):
                return False
    return True


# ---------------------------------------------------------------------------
# Stage 2 — scoring. Dumb, fast, exists so stage 3 never sees a bad candidate.
# ---------------------------------------------------------------------------

WEIGHTS = {"direction": 0.40, "values": 0.30, "complement": 0.15, "comm": 0.15}


def _trait_conf_pair(user: dict, cand: dict, trait: str) -> float:
    """0..1 weight: how much both sides actually know about this trait."""
    return min(user[trait]["confidence"], cand[trait]["confidence"]) / 100.0


def score_candidates(user: dict, candidates: list[dict]) -> list[tuple[float, dict]]:
    u_dir = user.get("direction") or ", ".join(user.get("values", ["nothing yet"]))
    u_val = ", ".join(user.get("values", [])) or "none"

    texts = [u_dir, u_val]
    for c in candidates:
        p = c["profile"]
        texts.append(p.get("direction") or ", ".join(p.get("values", ["nothing"])))
        texts.append(", ".join(p.get("values", [])) or "none")
    vecs = embed(texts)
    u_dir_v, u_val_v = vecs[0], vecs[1]

    scored = []
    for i, c in enumerate(candidates):
        p = c["profile"]
        c_dir_v, c_val_v = vecs[2 + 2 * i], vecs[3 + 2 * i]

        s_direction = float(u_dir_v @ c_dir_v)
        s_values = float(u_val_v @ c_val_v)

        # complementarity ONLY on energy + planning (research: similarity wins
        # on values/goals; complement has weak support outside these axes),
        # and only as strong as our confidence in both readings.
        s_comp = 0.0
        for trait in ("energy", "planning"):
            diff = abs(user[trait]["value"] - p[trait]["value"]) / 100.0
            s_comp += 0.5 * diff * _trait_conf_pair(user, p, trait)

        # communication style: similarity, confidence-weighted
        diff = abs(user["comm_depth"]["value"] - p["comm_depth"]["value"]) / 100.0
        s_comm = (1.0 - diff) * _trait_conf_pair(user, p, "comm_depth")

        total = (WEIGHTS["direction"] * s_direction + WEIGHTS["values"] * s_values
                 + WEIGHTS["complement"] * s_comp + WEIGHTS["comm"] * s_comm)
        scored.append((total, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Stage 3 — Gemma reads the finalists. Words in, words out.
# ---------------------------------------------------------------------------

def _traits_to_words(p: dict) -> str:
    out = []
    for trait, (low, high) in (("energy", ("solitary", "social")),
                               ("planning", ("improviser", "planner")),
                               ("openness", ("fixed views", "revises beliefs")),
                               ("comm_depth", ("light talker", "deep talker"))):
        t = p[trait]
        if t["confidence"] < 30:
            continue
        lean = low if t["value"] < 50 else high
        surety = "clearly" if t["confidence"] >= 70 else "leans"
        out.append(f"{surety} {lean}")
    return "; ".join(out) or "temperament unclear"


def _summarise(p: dict) -> str:
    parts = []
    if p.get("direction"):
        parts.append(f"direction: {p['direction']}")
    if p.get("values"):
        parts.append(f"values: {', '.join(p['values'][:6])}")
    parts.append(f"temperament: {_traits_to_words(p)}")
    if p.get("strengths"):
        parts.append(f"strengths: {', '.join(p['strengths'][:4])}")
    if p.get("gaps"):
        parts.append(f"gaps: {', '.join(p['gaps'][:4])}")
    if p.get("dealbreakers"):
        parts.append(f"avoids: {', '.join(p['dealbreakers'][:4])}")
    return " | ".join(parts)


RERANK_SYSTEM = """You are Orbit's matchmaker. You receive one USER and a numbered list
of CANDIDATES (all already passed intent and dealbreaker filters, all pre-scored
as plausible). Choose the 3 best people for the user to meet.

Connection types (choose the one that truly fits, not the fanciest):
  mirror         — alike in temperament and values; instant understanding
  complement     — one's strengths cover the other's gaps
  growth         — one is a step ahead on the path the other is walking
  opposite_world — different domains, same curiosity; expands both
  shared_journey — fighting the same battle at the same time

Rules:
1. NEVER select anyone who plausibly conflicts with the user's "avoids" list.
2. The reason must be concrete and reference BOTH people's actual details.
   Plain words. No percentages, no scores, no flattery.
3. first_activity: one specific thing they could do together in 30 minutes,
   tailored to THEM (not "grab coffee").
4. Use each candidate's number exactly as given.

Return JSON:
{"matches": [
  {"candidate": <number>,
   "connection_type": "mirror|complement|growth|opposite_world|shared_journey",
   "reason": "one or two sentences, plain and specific",
   "first_activity": "one concrete 30-minute activity"}
]}"""


def rerank(user: dict, top: list[tuple[float, dict]]) -> list[dict]:
    lines = [f"USER: {_summarise(user)}", "", "CANDIDATES:"]
    for i, (_, c) in enumerate(top, start=1):
        lines.append(f"{i}. {c['name']}: {_summarise(c['profile'])}")

    result = chat_json(RERANK_SYSTEM, "\n".join(lines))

    # guard: candidate numbers must be real, max 3 matches
    matches = []
    for m in result.get("matches", [])[:3]:
        idx = m.get("candidate")
        if isinstance(idx, int) and 1 <= idx <= len(top):
            m["name"] = top[idx - 1][1]["name"]
            m["profile"] = top[idx - 1][1]["profile"]
            matches.append(m)
    return matches


# ---------------------------------------------------------------------------
# The funnel, end to end.
# ---------------------------------------------------------------------------

def find_matches(user: dict, candidates: list[dict], shortlist: int = 12) -> list[dict]:
    survivors = [c for c in candidates if passes_hard_filters(user, c["profile"])]
    if not survivors:
        return []
    top = score_candidates(user, survivors)[:shortlist]
    return rerank(user, top)


if __name__ == "__main__":
    with open("me.json", encoding="utf-8") as f:
        user = json.load(f)["profile"]
    with open("test_profiles.json", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"pool: {len(candidates)} candidates")
    survivors = [c for c in candidates if passes_hard_filters(user, c["profile"])]
    print(f"after hard filters: {len(survivors)} "
          f"(removed: {[c['name'] for c in candidates if c not in survivors]})")

    scored = score_candidates(user, survivors)
    print("\nstage 2 ranking:")
    for s, c in scored:
        print(f"  {s:.3f}  {c['name']}")

    print("\nstage 3 — orbit's picks:\n")
    for m in rerank(user, scored[:12]):
        print(f"  ✦ {m['name']}  [{m['connection_type']}]")
        print(f"    {m['reason']}")
        print(f"    first 30 min: {m['first_activity']}\n")
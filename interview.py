"""
interview.py — Orbit's interview: seven questions, one profile.

Run it in the terminal tonight. The same loop becomes the FastAPI endpoint
tomorrow — the logic doesn't change, only the transport.

Key design decision: ALLOWED_FIELDS. Each question may only write the fields
it was designed to discover. Gemma sometimes infers energy from a project
description; the filter makes that structurally impossible instead of asking
the model nicely. (Deterministic guard #4.)
"""

import json

from extract import (EMPTY_PROFILE, extract_from_answer, extract_style,
                     merge_into_profile)

# ---------------------------------------------------------------------------
# The questions. Order follows effort: easy first, depth later.
# ---------------------------------------------------------------------------

QUESTIONS = [
    # (id, question text, fields this question is allowed to write)
    (1, "First things first — what are you hoping to find here? A collaborator "
        "for something you're building, a friend, or just someone to think out loud with?",
        {"intent"}),

    (2, "What are you actually spending your time on right now? And — honestly — "
        "what's the part of it you keep putting off?",
        {"direction", "gaps", "strengths"}),

    (3, "What's an idea you keep coming back to — something you could talk about "
        "for an hour without noticing?",
        {"values"}),

    (4, "Think of the last time plans actually got cancelled on you. What did you "
        "do with the evening — and honestly, how did you feel when you got the message?",
        {"energy", "planning"}),

    (5, "Tell me about a time you changed your mind about something that mattered to you.",
        {"openness"}),

    (6, "What kind of person exhausts you?",
        {"dealbreakers", "values"}),
]

# Q7 is adaptive: probe whatever we know least about.
ADAPTIVE_POOL = {
    "energy":    ("When you're completely drained, what actually recharges you?",
                  {"energy"}),
    "planning":  ("How did your last weekend actually come together — planned in "
                  "advance, or did it just happen?",
                  {"planning"}),
    "openness":  ("What's something you were completely wrong about?",
                  {"openness"}),
    "strengths": ("What do people usually come to you for?",
                  {"strengths"}),
}

# Fields any question may contribute, regardless of target
ALWAYS_ALLOWED = {"values", "evidence", "reflect_back"}


def _filter_extraction(extraction: dict, allowed: set) -> dict:
    """Deterministic guard #4: a question can only write what it was built to ask."""
    permitted = allowed | ALWAYS_ALLOWED
    filtered = {k: v for k, v in extraction.items() if k in permitted}
    # evidence entries must also point at permitted fields
    if "evidence" in filtered:
        filtered["evidence"] = {k: v for k, v in filtered["evidence"].items()
                                if k in permitted}
        if not filtered["evidence"]:
            del filtered["evidence"]
    return filtered


def _pick_adaptive(profile: dict) -> tuple[str, set]:
    """Choose Q7: the trait we're least confident about, or strengths if empty."""
    if not profile["strengths"]:
        return ADAPTIVE_POOL["strengths"]
    weakest = min(("energy", "planning", "openness"),
                  key=lambda t: profile[t]["confidence"])
    return ADAPTIVE_POOL[weakest]


def _print_graph(profile: dict) -> None:
    """Tiny terminal version of the right-hand panel."""
    print("\n  ── what orbit sees ──")
    for trait, (low, high) in (("energy", ("solitary", "social")),
                               ("planning", ("improviser", "planner")),
                               ("openness", ("fixed", "revising"))):
        p = profile[trait]
        if p["confidence"] == 0:
            print(f"  {trait:<9} not enough signal yet")
            continue
        lean = low if p["value"] < 50 else high
        bar = "▓" * (p["confidence"] // 10) + "░" * (10 - p["confidence"] // 10)
        print(f"  {trait:<9} {lean:<10} confidence {bar}")
    if profile["values"]:
        print(f"  values    {', '.join(profile['values'][:5])}")
    if profile["dealbreakers"]:
        print(f"  avoids    {', '.join(profile['dealbreakers'][:3])}")
    print()


def run_interview() -> dict:
    import copy
    profile = copy.deepcopy(EMPTY_PROFILE)
    transcript: list[tuple[str, str]] = []

    print("\n" + "=" * 60)
    print("  ORBIT — let's figure out who you actually are.")
    print("  (answer honestly; short is fine, stories are better)")
    print("=" * 60 + "\n")

    for qid, question, allowed in QUESTIONS:
        print(f"orbit: {question}\n")
        answer = input("you:   ").strip()
        transcript.append((question, answer))

        extraction = _filter_extraction(extract_from_answer(question, answer), allowed)
        profile = merge_into_profile(profile, extraction)

        if extraction.get("reflect_back"):
            print(f"\norbit: {extraction['reflect_back']}")
        _print_graph(profile)

    # Q7 — adaptive
    question, allowed = _pick_adaptive(profile)
    print(f"orbit: One more. {question}\n")
    answer = input("you:   ").strip()
    transcript.append((question, answer))
    extraction = _filter_extraction(extract_from_answer(question, answer), allowed)
    profile = merge_into_profile(profile, extraction)
    if extraction.get("reflect_back"):
        print(f"\norbit: {extraction['reflect_back']}")

    # comm_depth: measured from HOW they answered. Never asked.
    style = extract_style(transcript)
    profile = merge_into_profile(profile, style)

    _print_graph(profile)
    print("orbit: That's you — or at least, what your answers show. "
          "Anything on that panel look wrong, you'll be able to fix it.\n")

    return {"profile": profile, "transcript": transcript}


if __name__ == "__main__":
    result = run_interview()
    with open("me.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("saved -> me.json  (your profile + transcript; also tonight's test data)")
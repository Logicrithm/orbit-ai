"""
extract.py — Orbit's core: turn interview answers into a personality profile.

This is the most important file in the project. If this is good, Orbit works.

Design decisions, so future-you remembers why:
  - Every scored trait carries confidence. The AI admits what it doesn't know.
  - Every inference carries evidence: the user's own words that produced it.
    This is what we show on the right-hand panel, and it is the thesis made
    visible: behaviour over self-description.
  - Extraction runs per-answer (incremental, feeds the live graph) and the
    final profile is just the accumulated state plus one style pass over the
    whole transcript.
"""

from llm import chat_json

# ---------------------------------------------------------------------------
# The profile schema. One place, one truth.
# ---------------------------------------------------------------------------

EMPTY_PROFILE = {
    "intent": None,            # hard filter: collaborator | friend | sounding_board | unsure
    "dealbreakers": [],        # hard filter: list of short phrases
    "direction": "",           # free text: what they're building / moving toward
    "values": [],              # list of short phrases
    "strengths": [],
    "gaps": [],
    "energy":     {"value": 50, "confidence": 0},   # 0 solitary .. 100 social
    "planning":   {"value": 50, "confidence": 0},   # 0 improviser .. 100 planner
    "openness":   {"value": 50, "confidence": 0},   # 0 fixed .. 100 revising
    "comm_depth": {"value": 50, "confidence": 0},   # 0 light .. 100 deep  (measured, never asked)
    "evidence": {},            # trait -> short quote from the user's own words
}

# Which question feeds which fields (from the finalized question set)
QUESTION_TARGETS = {
    1: ["intent"],
    2: ["direction", "gaps"],
    3: ["values"],
    4: ["energy", "planning"],
    5: ["openness"],
    6: ["dealbreakers", "values"],
    7: [],  # adaptive — targets decided at runtime
}

EXTRACT_SYSTEM = """You are the analysis engine of Orbit, an app that infers who someone
actually is from their behaviour and stories — never from their self-description.

You receive one interview question and the person's answer. Extract ONLY what the
answer actually supports. Rules:

1. Behaviour over claims. "I stayed home and felt relieved" is evidence about energy.
   "I'm an introvert" is a self-label — note it but weight it low.
2. Confidence is honesty. A rich, specific story earns confidence 60-85.
   A vague or one-line answer earns 10-30. Never output confidence above 85
   from a single answer. If the answer contains nothing about a trait,
   do not mention that trait at all.
3. Evidence is a SHORT quote or close paraphrase of the person's own words —
   the exact phrase that justified your inference. Max 12 words.
4. Shadow reading: what exhausts or annoys someone reveals what they value.
   "People who never follow through drain me" implies they value follow-through.
5. If the answer is empty, evasive, or "idk": return low/zero confidence and
   move on. Never invent. An honest "not enough signal" is a correct output.

Return JSON with EXACTLY this shape (omit any key the answer gave no signal for):
{
  "intent": "collaborator" | "friend" | "sounding_board" | "unsure",
  "dealbreakers": ["short phrase", ...],
  "direction": "one sentence, their words where possible",
  "values": ["short phrase", ...],
  "strengths": ["short phrase", ...],
  "gaps": ["short phrase", ...],
  "energy":   {"value": 0-100, "confidence": 0-100},
  "planning": {"value": 0-100, "confidence": 0-100},
  "openness": {"value": 0-100, "confidence": 0-100},
  "evidence": {"trait_name": "their words, max 12 words"},
  "reflect_back": "One short sentence telling the person something true you just
                   learned about them. Warm, specific, no flattery. This is shown
                   to them immediately."
}"""


def extract_from_answer(question: str, answer: str) -> dict:
    """One answer -> one incremental extraction. Returns {} for empty input."""
    if not answer or not answer.strip():
        return {}
    user = f"QUESTION ASKED:\n{question}\n\nTHEIR ANSWER:\n{answer}"
    return chat_json(EXTRACT_SYSTEM, user)


def merge_into_profile(profile: dict, extraction: dict) -> dict:
    """
    Fold one extraction into the running profile.

    Merge rules:
      - lists: append new items, no duplicates
      - scored traits: keep whichever reading has higher confidence
        (simple and defensible; a weighted average would blur evidence)
      - intent/direction: latest non-empty wins
      - evidence: accumulate
    """
    for key in ("values", "strengths", "gaps", "dealbreakers"):
        for item in extraction.get(key, []):
            if item and item.lower() not in [x.lower() for x in profile[key]]:
                profile[key].append(item)

    for trait in ("energy", "planning", "openness", "comm_depth"):
        new = extraction.get(trait)
        if new and new.get("confidence", 0) > profile[trait]["confidence"]:
            profile[trait] = {"value": new["value"], "confidence": new["confidence"]}

    if extraction.get("intent"):
        profile["intent"] = extraction["intent"]
    if extraction.get("direction"):
        profile["direction"] = extraction["direction"]

    profile["evidence"].update(extraction.get("evidence", {}))
    return profile


# ---------------------------------------------------------------------------
# comm_depth: measured from HOW they answered, never asked.
# ---------------------------------------------------------------------------

STYLE_SYSTEM = """You analyze HOW a person communicated across an interview transcript —
not what they said, but how. Signals: answer length and specificity, whether they
volunteer feelings, whether they reflect or deflect, story-telling vs bullet-point
minimalism.

Return JSON:
{
  "comm_depth": {"value": 0-100, "confidence": 0-100},
  "evidence": {"comm_depth": "observation in max 12 words, e.g. 'long reflective answers, volunteers feelings unprompted'"}
}
0 = brief, light, transactional. 100 = long-form, reflective, depth-seeking.
Confidence: 6-7 real answers earns 50-75. Thin transcript earns less."""


def extract_style(transcript: list[tuple[str, str]]) -> dict:
    """transcript: list of (question, answer). One pass over the whole thing."""
    text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in transcript)
    return chat_json(STYLE_SYSTEM, text)


# ---------------------------------------------------------------------------
# Torture test — run this file directly. Ugly inputs on purpose.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tests = [
        ("Think of the last time plans got cancelled on you. What did you do, and how did you feel?",
         "honestly? relieved lol. ordered biryani, put on a video essay about roman concrete, "
         "worked on my side project till 2am. best friday in weeks"),
        ("Think of the last time plans got cancelled on you. What did you do, and how did you feel?",
         "idk"),
        ("What kind of person exhausts you?",
         "people who talk in circles in meetings and never actually DO anything. like just ship "
         "something?? also people who make everything about status"),
        ("What are you actually spending your time on right now?",
         "uh a lot of stuff i guess. college, some coding, normal things"),
    ]

    for q, a in tests:
        print("=" * 70)
        print("Q:", q)
        print("A:", a)
        result = extract_from_answer(q, a)
        print(json.dumps(result, indent=2))

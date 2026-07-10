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

You receive one interview question and the person's answer.

THE MOST IMPORTANT RULE: only output a field if the answer DIRECTLY supports it.
The schema is a menu, not a form. Most answers support only 1-3 fields.
Omitting a field is correct. Inventing a value for it is the worst possible error.

For the three temperament traits, do NOT output numbers. Name what you saw:

  "energy":   {"leaning": "solitary" or "social",    "strength": "hint" or "clear"}
  "planning": {"leaning": "improviser" or "planner", "strength": "hint" or "clear"}
  "openness": {"leaning": "fixed" or "revising",     "strength": "hint" or "clear"}

  solitary  = recharges alone (stayed in, relieved, solo work)
  social    = recharges with people (called friends, went out, energised by company)
  improviser= decides in the moment, goes with the flow
  planner   = backup plans, schedules, structure
  fixed     = holds positions firmly ; revising = updates beliefs readily

  strength "clear" = a specific story or behaviour shows it directly
  strength "hint"  = weak signal, indirect, OR a bare self-label with no story
                     ("I'm an introvert / organised / open-minded" with nothing
                     behind it is ALWAYS at most a hint — labels are claims,
                     behaviour is evidence)
  no signal        = omit the trait entirely

RULES:
1. Behaviour over claims, always.
2. Evidence keys must be the ACTUAL trait name, value is the person's own words,
   max 12 words. Example: "evidence": {"energy": "stayed in, felt relieved"}
3. Shadow reading: what exhausts someone reveals what they value.
   "People who never follow through drain me" → values: ["follow-through"].
4. values = principles ("action over talk", "depth"), NOT activities
   (not "coding", not "college").
5. strengths and gaps ONLY if the person literally described something they are
   good at, bad at, or keep putting off. Never infer them from tone.
6. direction ONLY if the answer describes what they're building or moving toward.
   Otherwise omit the key entirely.

Return JSON (omit any key without direct support):
{
  "intent": "collaborator" | "friend" | "sounding_board" | "unsure",
  "dealbreakers": ["short phrase", ...],
  "direction": "one sentence, their words where possible",
  "values": ["short phrase", ...],
  "strengths": ["short phrase", ...],
  "gaps": ["short phrase", ...],
  "energy":   {"leaning": "...", "strength": "..."},
  "planning": {"leaning": "...", "strength": "..."},
  "openness": {"leaning": "...", "strength": "..."},
  "evidence": {"energy": "their words, max 12 words"},
  "reflect_back": "One short sentence telling the person something true you just
                   learned about them. Warm, specific, no flattery."
}"""

# ---------------------------------------------------------------------------
# Leaning -> number happens HERE, deterministically. A 4B model coin-flips
# bipolar numeric scales; it does not coin-flip the word "solitary".
# ---------------------------------------------------------------------------

_TRAIT_POLES = {
    # trait: (low-end leaning, high-end leaning)
    "energy":     ("solitary", "social"),
    "planning":   ("improviser", "planner"),
    "openness":   ("fixed", "revising"),
    "comm_depth": ("light", "deep"),
}
_STRENGTH_MAP = {
    # strength: (distance from 50, confidence)
    "clear": (35, 75),   # -> value 15 or 85
    "hint":  (15, 35),   # -> value 35 or 65
}


def _leanings_to_numbers(extraction: dict) -> dict:
    """Convert categorical trait readings to the numeric profile schema in place."""
    for trait, (low, high) in _TRAIT_POLES.items():
        reading = extraction.get(trait)
        if not isinstance(reading, dict) or "leaning" not in reading:
            extraction.pop(trait, None)
            continue
        leaning = str(reading.get("leaning", "")).lower()
        strength = str(reading.get("strength", "hint")).lower()
        dist, conf = _STRENGTH_MAP.get(strength, _STRENGTH_MAP["hint"])
        if leaning == low:
            extraction[trait] = {"value": 50 - dist, "confidence": conf}
        elif leaning == high:
            extraction[trait] = {"value": 50 + dist, "confidence": conf}
        else:
            extraction.pop(trait, None)   # unknown leaning: no signal
    return extraction


VALID_INTENTS = {"collaborator", "friend", "sounding_board", "unsure"}


def _guard(extraction: dict, answer: str) -> dict:
    """
    Deterministic guards. No prompt can make a 4B model fully obedient,
    so code enforces what the prompt requests:

    1. Evidence-gated confidence: a trait keeps 'clear' confidence (75) only if
       its evidence quote is actually grounded in the answer text. Fabricated
       quote -> evidence dropped, confidence downgraded to hint (35).
       The model must cite the person's real words to earn certainty.
    2. intent must be in the enum, else 'unsure'.
    3. Empty strings/lists are noise -> removed.
    """
    answer_l = answer.lower()
    evidence = extraction.get("evidence", {}) or {}

    # 1. verify each evidence quote is substantially present in the answer
    for trait, quote in list(evidence.items()):
        words = [w for w in str(quote).lower().split() if len(w) > 3]
        if not words:
            grounded = False
        else:
            grounded = sum(1 for w in words if w in answer_l) / len(words) >= 0.5
        if not grounded:
            del evidence[trait]
            if trait in _TRAIT_POLES and isinstance(extraction.get(trait), dict):
                if extraction[trait].get("confidence", 0) > 35:
                    extraction[trait]["confidence"] = 35

    # any clear-confidence trait with NO evidence at all also drops to hint
    for trait in _TRAIT_POLES:
        t = extraction.get(trait)
        if isinstance(t, dict) and t.get("confidence", 0) > 35 and trait not in evidence:
            t["confidence"] = 35

    # 2. enum guard
    if extraction.get("intent") not in VALID_INTENTS:
        extraction.pop("intent", None)

    # 3. drop empty noise
    for k in list(extraction.keys()):
        if extraction[k] in ("", [], {}, None):
            del extraction[k]

    return extraction


def extract_from_answer(question: str, answer: str) -> dict:
    """One answer -> one incremental extraction. Returns {} for empty input.
    Trait leanings are converted to numbers here, so everything downstream
    (merge, matching, UI) still sees {"value": n, "confidence": n}."""
    if not answer or not answer.strip():
        return {}
    user = f"QUESTION ASKED:\n{question}\n\nTHEIR ANSWER:\n{answer}"
    return _guard(_leanings_to_numbers(chat_json(EXTRACT_SYSTEM, user)), answer)


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
        profile[key] = profile[key][:6]

    for trait in ("energy", "planning", "openness", "comm_depth"):
        new = extraction.get(trait)
        if new and new.get("confidence", 0) > profile[trait]["confidence"]:
            profile[trait] = {"value": new["value"], "confidence": new["confidence"]}

    if extraction.get("intent"):
        profile["intent"] = extraction["intent"]
    if extraction.get("direction"):
        profile["direction"] = extraction["direction"]

    ev = extraction.get("evidence", {})
    if isinstance(ev, dict):
        profile["evidence"].update(ev)
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
  "comm_depth": {"leaning": "light" or "deep", "strength": "hint" or "clear"},
  "evidence": {"comm_depth": "observation in max 12 words, e.g. 'long reflective answers, volunteers feelings unprompted'"}
}
light = brief, transactional answers. deep = long-form, reflective, depth-seeking.
strength "clear" needs 4+ substantial answers; a thin transcript is at most a hint."""


def extract_style(transcript: list[tuple[str, str]]) -> dict:
    """transcript: list of (question, answer). One pass over the whole thing."""
    text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in transcript)
    return _leanings_to_numbers(chat_json(STYLE_SYSTEM, text))


# ---------------------------------------------------------------------------
# Torture test — run this file directly. Ugly inputs on purpose.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Held-out tests. None of these appear in any prompt. (Standing rule.)
    tests = [
        # solitary story, new content — does "solitary" come out this time?
        ("Think of the last time plans got cancelled on you. What did you do, and how did you feel?",
         "ngl kind of happy about it. made chai, reorganized my desk, got 3 hours of "
         "uninterrupted reading. texted 'no worries!' maybe too fast lol"),

        # improviser story — planning low end, never shown as an example
        ("Think of the last time plans got cancelled on you. What did you do, and how did you feel?",
         "just walked out the door with no destination, found some street food place, "
         "got talking to the owner for an hour. best evenings are the unplanned ones"),

        # self-label trap, DIFFERENT label than last round — planning this time
        ("Think of the last time plans got cancelled on you. What did you do, and how did you feel?",
         "well i'm an extremely organized person, everyone says that about me, so you can imagine"),

        # genuine direction answer — does direction fill correctly when it should?
        ("What are you actually spending your time on right now?",
         "building a wifi sensing project — using signal data to detect human activity "
         "without cameras. the part i keep putting off is writing the documentation"),
    ]

    for q, a in tests:
        print("=" * 70)
        print("Q:", q)
        print("A:", a)
        result = extract_from_answer(q, a)
        print(json.dumps(result, indent=2))
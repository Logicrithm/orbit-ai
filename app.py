"""
app.py — Orbit's backend. Wraps the already-tested logic; adds no new logic.

Run:  uvicorn app:app --reload
Then open http://localhost:8000

Sessions live in memory — this is a demo, not a deployment. The honest-limits
section of the README says so.
"""

import copy
import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from extract import EMPTY_PROFILE, extract_from_answer, extract_style, merge_into_profile
from interview import QUESTIONS, ADAPTIVE_POOL, ALWAYS_ALLOWED, _filter_extraction, _pick_adaptive
from match import find_matches

app = FastAPI(title="Orbit")

SESSIONS: dict[str, dict] = {}

with open("seeds.json", encoding="utf-8") as f:
    POPULATION = json.load(f)
print(f"[orbit] population loaded: {len(POPULATION)} profiles")

# warm the embedder at startup, not on the first match click
from match import embed
embed(["warmup"])
print("[orbit] embedder ready")


class AnswerIn(BaseModel):
    session_id: str
    answer: str


class EditIn(BaseModel):
    session_id: str
    action: str          # "flip_trait" | "remove_item"
    field: str           # trait name, or list name
    value: str = ""      # item text for remove_item


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/api/start")
def start():
    sid = str(uuid.uuid4())[:8]
    SESSIONS[sid] = {
        "profile": copy.deepcopy(EMPTY_PROFILE),
        "transcript": [],
        "q_index": 0,          # 0..5 fixed, 6 adaptive, 7 done
        "adaptive": None,
    }
    return {"session_id": sid, "question": QUESTIONS[0][1], "number": 1, "total": 7}


@app.post("/api/answer")
def answer(inp: AnswerIn):
    s = SESSIONS.get(inp.session_id)
    if not s:
        raise HTTPException(404, "unknown session")
    if s["q_index"] >= 7:
        raise HTTPException(400, "interview already complete")

    # which question was just answered?
    if s["q_index"] < 6:
        _, question, allowed = QUESTIONS[s["q_index"]]
    else:
        question, allowed = s["adaptive"]

    s["transcript"].append((question, inp.answer))
    extraction = _filter_extraction(extract_from_answer(question, inp.answer), allowed)
    s["profile"] = merge_into_profile(s["profile"], extraction)
    s["q_index"] += 1

    done = s["q_index"] >= 7
    next_q, number = None, None
    if not done:
        if s["q_index"] < 6:
            next_q = QUESTIONS[s["q_index"]][1]
        else:
            s["adaptive"] = _pick_adaptive(s["profile"])
            next_q = "One more. " + s["adaptive"][0]
        number = s["q_index"] + 1

    if done:
        # comm_depth: measured from HOW they answered, never asked
        s["profile"] = merge_into_profile(s["profile"], extract_style(s["transcript"]))

    return {
        "reflect_back": extraction.get("reflect_back", ""),
        "profile": s["profile"],
        "done": done,
        "question": next_q,
        "number": number,
        "total": 7,
    }


@app.post("/api/edit")
def edit(inp: EditIn):
    """The 'that's wrong' button. The user is the final authority on themselves."""
    s = SESSIONS.get(inp.session_id)
    if not s:
        raise HTTPException(404, "unknown session")
    p = s["profile"]

    if inp.action == "flip_trait" and inp.field in ("energy", "planning", "openness", "comm_depth"):
        t = p[inp.field]
        t["value"] = 100 - t["value"]
        t["confidence"] = 90          # user said so; that outranks any inference
        p["evidence"][inp.field] = "corrected by user"
    elif inp.action == "remove_item" and inp.field in ("values", "strengths", "gaps", "dealbreakers"):
        p[inp.field] = [x for x in p[inp.field] if x != inp.value]
    else:
        raise HTTPException(400, "unknown edit")
    return {"profile": p}


@app.post("/api/match")
def match(inp: AnswerIn):   # reuses AnswerIn; only session_id is read
    s = SESSIONS.get(inp.session_id)
    if not s:
        raise HTTPException(404, "unknown session")
    matches = find_matches(s["profile"], POPULATION)
    s["matches"] = {m["name"]: m for m in matches}   # kept for the intro room
    s["chats"] = {}
    out = [{"name": m["name"], "connection_type": m["connection_type"],
            "reason": m["reason"], "first_activity": m["first_activity"]}
           for m in matches]
    return {"matches": out}


class IntroIn(BaseModel):
    session_id: str
    name: str


class ChatIn(BaseModel):
    session_id: str
    name: str
    message: str


from llm import chat
from match import _summarise

INTRO_SYSTEM = """You are Orbit introducing two people who both accepted a match.
Write like a warm mutual friend making an introduction: 2-3 sentences, name the real
reason they fit (from their profiles only — invent nothing), end by pointing them at
their first activity.
RULES:
- Output ONLY the introduction itself. No preamble like "here's the introduction",
  no quotation marks around it.
- Person B's name is given. Person A's name is NOT known: address them as "you" —
  NEVER invent a name for them.
- Plain words, no flattery."""

PERSONA_CHAT_SYSTEM = """You are role-playing ONE specific person (your name and profile
are given) in their first chat with a new match. Reply like a real person texting:
1-3 casual sentences, warm but natural, referencing your own life from your profile.
RULES:
- You are the named person. The OTHER person's name is unknown — never invent a name
  for them, and NEVER greet them with your own name.
- Never mention being an AI or simulated."""


@app.post("/api/intro")
def intro(inp: IntroIn):
    s = SESSIONS.get(inp.session_id)
    if not s or inp.name not in s.get("matches", {}):
        raise HTTPException(404, "unknown match")
    m = s["matches"][inp.name]
    user_txt = f"PERSON A (the user): {_summarise(s['profile'])}"
    cand_txt = f"PERSON B ({inp.name}): {_summarise(m['profile'])}"
    act = f"Their first activity: {m['first_activity']}"
    intro_msg = chat(INTRO_SYSTEM, f"{user_txt}\n\n{cand_txt}\n\n{act}")

    opener = chat(PERSONA_CHAT_SYSTEM,
                  f"YOUR NAME: {inp.name}\nYOUR PROFILE: {_summarise(m['profile'])}\n"
                  f"You just matched with someone (their name is unknown to you). "
                  f"Orbit's intro said: {intro_msg}\n"
                  f"Send your first message to them:")
    s["chats"][inp.name] = [("them", opener)]
    return {"intro": intro_msg, "opener": opener}


@app.post("/api/chat")
def persona_chat(inp: ChatIn):
    s = SESSIONS.get(inp.session_id)
    if not s or inp.name not in s.get("matches", {}):
        raise HTTPException(404, "unknown match")
    hist = s["chats"].setdefault(inp.name, [])
    hist.append(("you", inp.message))
    convo = "\n".join(f"{'THEM' if who == 'you' else 'YOU'}: {msg}" for who, msg in hist[-8:])
    reply = chat(PERSONA_CHAT_SYSTEM,
                 f"YOUR NAME: {inp.name}\n"
                 f"YOUR PROFILE: {_summarise(s['matches'][inp.name]['profile'])}\n"
                 f"CONVERSATION SO FAR:\n{convo}\n\nYour reply:")
    hist.append(("them", reply))
    return {"reply": reply}
"""
seed.py — Orbit's population generator.

Usage:
    python seed.py --count 30                    # tonight, on Ollama
    python seed.py --count 150 --out seeds.json  # on the MI300X, same command

How it works, and why this way:
  1. 20 hand-written PERSONA SEEDS spread across intent x temperament x domain
     x life-stage. Deliberate spread — a lazily-generated population collapses
     into 150 copies of the same person and every match demo looks alike.
  2. Gemma ANSWERS THE INTERVIEW as each persona (with noise: each gets a
     random name, age band, and a "quirk" so two profiles from one seed differ).
  3. The answers go through the REAL extraction pipeline — extract_from_answer,
     the guards, the per-question filters, merge. Same code path as a live user.
     Every profile in the database was born the way a real one would be.

Contamination rule (standing): no persona below resembles the hand-made test
profiles or any real user profile. These test the pipeline, not flatter it.

Prints throughput stats at the end — on the MI300X run, those numbers go
straight into the README's ### Workload section.
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

from extract import EMPTY_PROFILE, extract_from_answer, extract_style, merge_into_profile
from interview import QUESTIONS, ALWAYS_ALLOWED, _filter_extraction
from llm import chat

# ---------------------------------------------------------------------------
# 20 persona seeds. Spread, not similarity.
# ---------------------------------------------------------------------------

PERSONAS = [
    # (short label, description Gemma will role-play)
    ("startup-grinder", "a 20-something building a B2B SaaS startup, obsessed with execution, "
     "impatient with theory, recharges by shipping features alone at night"),
    ("phd-biologist", "a PhD student in marine biology who loves slow, careful fieldwork, "
     "plans everything, wants a friend outside academia to feel human again"),
    ("standup-writer", "an aspiring stand-up comedian who works a boring bank job, processes "
     "life by joking about it, secretly very observant about people"),
    ("returning-mom", "a mother re-entering software work after a career break, methodical, "
     "underconfident but highly skilled, looking for a collaborator on small projects"),
    ("chess-teacher", "a school chess coach who thinks in systems, prefers one deep friendship "
     "to ten shallow ones, suspicious of small talk"),
    ("travel-nurse", "a nurse who moves cities every six months, makes friends fast and loses "
     "them faster, wants something that survives distance"),
    ("open-source-dev", "a self-taught programmer who maintains a small open-source library, "
     "communicates better in writing than speech, wants a collaborator"),
    ("dance-choreographer", "a choreographer who improvises everything, allergic to plans, "
     "energised by rooms full of people, wants a sounding board for a studio idea"),
    ("retired-army", "a recently retired army officer adjusting to civilian slowness, values "
     "reliability above all, blunt but warm, looking for a friend"),
    ("poetry-editor", "an editor at a small poetry press, listens more than talks, changed "
     "careers from law and never looked back, deep talker"),
    ("robotics-undergrad", "a second-year robotics student who builds fighting robots, "
     "competitive, plans builds meticulously, wants a collaborator for competitions"),
    ("cafe-owner", "a cafe owner who knows every regular's story, extroverted but exhausted "
     "by performative networking, wants real conversation"),
    ("climate-analyst", "a climate data analyst who argues with strangers about nuclear power, "
     "revises beliefs when shown data, wants a sparring partner for ideas"),
    ("game-artist", "a freelance game artist who works nights, shy in person and vivid online, "
     "wants a friend who understands irregular schedules"),
    ("med-resident", "an exhausted medical resident with 4 free hours a week, needs efficient "
     "meaningful connection, zero patience for flakiness"),
    ("street-photographer", "a street photographer who wanders without destination, spontaneous, "
     "collects strangers' stories, sounding board for a photo-book idea"),
    ("math-olympiad", "a maths olympiad trainer who finds people harder than proofs, wants a "
     "friend, deeply loyal once trust forms"),
    ("ngo-organizer", "an NGO field organizer who plans logistics for a living but craves "
     "spontaneity in personal life, energised by groups"),
    ("audiobook-narrator", "an audiobook narrator who spends all day alone in a booth with "
     "other people's words, wants conversation that goes somewhere"),
    ("quant-dropout", "a former quant who quit finance to teach high-school maths, took a 90% "
     "pay cut and is happier, wants a sounding board about what money is for"),
]

FIRST_NAMES = ["Aarav", "Ananya", "Vikram", "Priya", "Rohit", "Sneha", "Karan", "Divya",
               "Nikhil", "Pooja", "Amit", "Kavya", "Rahul", "Isha", "Sanjay", "Neha",
               "Farhan", "Tara", "Joseph", "Lakshmi", "Zoya", "Aditya", "Mira", "Dhruv"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45+"]
QUIRKS = ["is having an unusually honest day", "is slightly tired and gives shorter answers",
          "is in a good mood and rambles a bit", "answers carefully, like they rehearsed",
          "keeps making small self-deprecating jokes"]

ROLEPLAY_SYSTEM = """You are role-playing ONE specific person answering an interview question
for a friend-finding app. Stay fully in character. Answer the way a real person types:
casual, concrete, 1-4 sentences, specific details from your life. Never mention being
an AI or a persona. Never analyse yourself in trait language — just answer like a human."""


def persona_answers(persona_desc: str, quirk: str) -> list[str]:
    """Gemma answers the six fixed interview questions as this persona."""
    answers = []
    for _, question, _ in QUESTIONS:
        user = (f"YOU ARE: {persona_desc}. Today you {quirk}.\n\n"
                f"INTERVIEW QUESTION: {question}\n\nYour answer:")
        answers.append(chat(ROLEPLAY_SYSTEM, user, temperature=0.9))
    return answers


def build_profile(answers: list[str]) -> dict:
    """Run persona answers through the REAL pipeline — same path as a live user."""
    import copy
    profile = copy.deepcopy(EMPTY_PROFILE)
    transcript = []
    for (qid, question, allowed), answer in zip(QUESTIONS, answers):
        transcript.append((question, answer))
        extraction = _filter_extraction(extract_from_answer(question, answer), allowed)
        profile = merge_into_profile(profile, extraction)
    profile = merge_into_profile(profile, extract_style(transcript))
    return profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", default="seeds.json")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    random.seed()  # different names each run
    population, t0 = [], time.time()

    def make_one(i: int):
        label, desc = PERSONAS[i % len(PERSONAS)]
        name = random.choice(FIRST_NAMES)
        age = random.choice(AGE_BANDS)
        quirk = random.choice(QUIRKS)
        try:
            answers = persona_answers(desc, quirk)
            profile = build_profile(answers)
            print(f"  done: {name} ({label}, {age})")
            return {"id": f"s{i+1}", "name": name, "age_band": age,
                    "persona": label, "profile": profile}
        except Exception as e:
            print(f"  FAILED {label}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(make_one, range(args.count)):
            if result:
                population.append(result)
            if len(population) % 10 == 0:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(population, f, indent=2, ensure_ascii=False)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(population, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    print(f"  generated : {len(population)} profiles -> {args.out}")
    print(f"  total time: {elapsed/60:.1f} min  ({elapsed/max(len(population),1):.1f}s per profile)")
    print("  (on the MI300X run, these numbers -> README ### Workload)")


if __name__ == "__main__":
    main()
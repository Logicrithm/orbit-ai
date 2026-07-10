"""
llm.py — Orbit's single door to the model.

Every LLM call in Orbit goes through this file. Nothing else imports openai.
Why: tonight this points at Ollama (free). Saturday it points at the MI300X
tunnel. Nothing else in the codebase changes.

.env for tonight (Ollama):
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_API_KEY=ollama
    LLM_MODEL=gemma3:4b

.env for Saturday (MI300X via SSH tunnel):
    LLM_BASE_URL=http://localhost:8000/v1
    LLM_API_KEY=anything
    LLM_MODEL=google/gemma-3-4b-it
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)
MODEL = os.environ["LLM_MODEL"]


def chat(system: str, user: str, temperature: float = 0.7) -> str:
    """Plain text response. Used for interview questions and reflect-backs."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=1000,
    )
    return r.choices[0].message.content.strip()


def chat_json(system: str, user: str, temperature: float = 0.2) -> dict:
    """
    JSON response, guaranteed dict or raised error.

    Three layers of defence against a 4B model drifting out of format:
      1. json_object response format (Ollama and vLLM both support this)
      2. strip markdown fences if the model adds them anyway
      3. one retry with the parse error shown to the model
    Low temperature: extraction should be boring and repeatable.
    """
    messages = [
        {"role": "system", "content": system + "\nRespond with valid JSON only. No markdown, no explanation."},
        {"role": "user", "content": user},
    ]

    for attempt in range(2):
        r = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        raw = r.choices[0].message.content.strip()

        # Layer 2: strip fences some models add despite json mode
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt == 0:
                # Layer 3: show the model its own mistake, once
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": f"That was not valid JSON ({e}). Return ONLY the corrected JSON."})
            else:
                raise RuntimeError(f"Model returned unparseable JSON twice. Last output:\n{raw}") from e


if __name__ == "__main__":
    # Smoke test both doors
    print("chat():", chat("You are terse.", "Say hello in five words."))
    print("chat_json():", chat_json("You output JSON.", 'Return {"status": "ok", "n": 42}'))

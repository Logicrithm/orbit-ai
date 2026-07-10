# Orbit

**An AI that learns who you actually are — and finds the people you should meet.**

Built solo in 48 hours for the AMD Developer Hackathon: Act II (Track 3 — Unicorn).

---

## The problem

Every social app matches you on what you *claim* to be — bios, hobby checklists,
your best photo. But people describe their ideal self, not their actual self. So
the matches are made between two fictional people, and that is why they die after
"hey."

Loneliness is now studied as a public-health problem, and the people it hits
hardest are exactly the people bio-driven apps serve worst: the ones who can't
initiate, who don't perform well in profiles, who are quiet but deep.

**Social apps reward people who present well. Orbit works for people who don't.**

## What Orbit does

1. **Interviews you** — seven conversational questions about what you *did*, not
   what you're like. Behaviour over self-description.
2. **Shows its reasoning** — a live panel of what it inferred, each trait backed
   by *your own words* as evidence, each with an honest confidence level.
3. **Takes correction** — anything wrong, you flip or delete with one click. The
   user is the final authority on themselves. We call the idea
   **Proof of Personality**: not who you claim to be — who your choices show you are.
4. **Matches you** — against people who fit you, which is not always people *like*
   you. Five connection types: mirror, complement, growth, opposite-world,
   shared journey.
5. **Explains, then introduces** — never a percentage. A person, a connection type,
   a plain-words reason, and a concrete first activity. On mutual acceptance,
   Orbit writes the introduction itself — the AI as mutual friend.

## Running on AMD Instinct MI300X

All LLM inference runs on a single AMD Instinct MI300X (AMD Developer Cloud),
serving **Gemma 3 4B** through **vLLM 0.23.0** on **ROCm 7.2.4**. No external
LLM API is used at any point.

This is a design requirement, not a convenience: the interview collects intimate
personal detail, and that data never leaves our own infrastructure.
**Privacy by hardware, not by policy.**

### GPU

```
========================= ROCm System Management Interface =========================
Device  Node  IDs (DID, GUID)   Temp     Power    Partitions      SCLK    MCLK    VRAM%  GPU%
0       1     0x74b5, 21947     40.0°C   155.0W   NPS1, SPX, 0    139Mhz  900Mhz  0%     0%
```

![rocm-smi on the MI300X](evidence/rocm-smi.png)

### ROCm inference backend

Excerpts from [`evidence/vllm.log`](evidence/vllm.log):

```
[model.py:611]       Resolved architecture: Gemma3ForConditionalGeneration
[rocm.py:637]        Using Flash Attention backend for ViT model.
[rocm.py:583]        Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
                     Overriding with ROCM_ATTN out of potential backends:
                     ['ROCM_ATTN', 'ROCM_AITER_UNIFIED_ATTN', 'TRITON_ATTN'].
[activation.py:728]  [ROCm] PyTorch's native GELU with tanh approximation is unstable.
[gpu_worker.py:480]  Available KV cache memory: 162.84 GiB
[kv_cache_utils.py]  GPU KV cache size: 1,217,591 tokens
[kv_cache_utils.py]  Maximum concurrency for 8,192 tokens per request: 148.63x
[api_server.py:583]  Starting vLLM server on http://0.0.0.0:8000
```

Model load: 8.58 GiB, 12.4 seconds.

> Note: vLLM logs `device_config=cuda` and "Capturing CUDA graphs" even on ROCm —
> HIP mirrors the CUDA API surface, so the naming persists. The `[rocm.py]`
> backend-selection lines and the 162 GiB KV cache confirm the AMD device.

### Workload

The seeded population — 148 synthetic users — was generated end-to-end on the
MI300X: Gemma role-plays each persona through the full seven-question interview,
and every answer runs through the same extraction pipeline as a real user.

| | AMD MI300X (30 concurrent) | Consumer laptop (Ollama, sequential) |
|---|---|---|
| Per profile (13 LLM calls) | **0.8 s** | ~16 min |
| 148 profiles | **2.1 min** | ~40 hours (projected) |

Each profile requires 13 LLM calls (6 role-played answers + 7 extraction/analysis
passes). The MI300X's 148x concurrency is what makes the population — and the
product — feasible.

### Reproduce

Full logs in [`evidence/`](evidence/). Droplet setup: [`setup.md`](setup.md).
The serving stack rebuilds from scratch in ~10 minutes.

## How it works

```
 user answers (7 questions, plain language)
        │
        ▼
 Gemma 3 4B — extraction            categorical leanings, not numbers:
        │                           a 4B model coin-flips bipolar numeric
        ▼                           scales; it does not coin-flip "solitary"
 deterministic guards (code)
   • evidence-gating: a trait keeps high confidence ONLY if its evidence
     quote actually appears in the user's answer — no receipt, no certainty
   • per-question field filters: a question can only write the traits it
     was designed to discover
   • enum + schema validation
        │
        ▼
 profile  { intent, values, dealbreakers, direction, strengths, gaps,
            energy / planning / openness / comm_depth  (value + confidence),
            evidence: the user's own words per trait }
        │
        ▼
 matching funnel
   1. hard filters (code)   — intent, dealbreakers. never weighted, never overridden
   2. scoring (numpy)       — direction embeddings (MiniLM, local CPU) + values
                              similarity + confidence-weighted complementarity
                              (energy/planning axes only) + communication fit
   3. rerank (Gemma)        — reads top candidates as WORDS, picks 3, names the
                              connection type, writes the reason + first activity
        │
        ▼
 mutual accept → Orbit writes the introduction → shared space
```

Design principle throughout: **numbers never enter or leave the model.** Profiles
are converted to words before Gemma sees them, and Gemma's words are converted to
numbers by code. The model does what it's good at (reading people); the code does
what it's good at (being exact). A small model's honesty is enforced, not requested.

One trait — `comm_depth` — is never asked about at all. It is measured from *how*
the person answered across the transcript: length, specificity, whether they
volunteer feelings. Asking "do you like deep conversations?" gets aspiration;
measuring gets behaviour.

## Safety and trust

Built into the demo:
- **18+ by design intent**; the product is for adults
- **Double opt-in** — no contact opens until both people accept; unwanted messages
  are impossible by construction
- **Dealbreakers are hard filters** — never soft-weighted, never overridden by a
  good score
- **Transparency panel** — the user sees every inference, its evidence, and can
  correct or delete anything; profile deletion removes the data

Designed, not yet built (production requirements):
- Phone-number verification (identity is a cost problem, not a detection problem)
- Report → human-review queue → graduated penalties; automated classifiers flag,
  humans decide
- Raw transcripts deleted after profile extraction; only derived traits persist

## What is and isn't demonstrated — honest limits

- **The population is synthetic.** 148 Gemma-role-played personas, generated from
  20 hand-written seeds. They test the pipeline; they do not prove real-world
  match quality. In the demo's shared space, the other person's acceptance and
  replies are simulated — and labeled as such on screen.
- **The matching weights are heuristics**, not validated coefficients. The staged
  design (filters → score → rerank) exists so weights can be tuned against real
  feedback later.
- **The feedback loop is designed, not built.** "Orbit asks how it went and gets
  better at knowing you" is the roadmap's core; nothing longitudinal is
  demonstrated in 48 hours.
- **Extraction has known soft spots**: occasional over-eager list items and
  vague `direction` summaries. The evidence-gating guard caps the damage, and
  the edit panel exists precisely because an AI reading seven answers will
  sometimes misread a human.
- **Seven questions of self-report, however behavioural, is still self-report.**
  Stories are harder to fake than trait claims — not impossible. Orbit is a
  better prior, not a truth machine.

## Roadmap

double opt-in ✅ → AI-brokered introduction ✅ → dedicated shared spaces →
post-interaction feedback → profile evolution → real-population pilot

## Run it yourself

```bash
pip install -r requirements.txt

# point .env at any OpenAI-compatible endpoint:
#   AMD MI300X + vLLM (see setup.md), or Ollama locally:
#     LLM_BASE_URL=http://localhost:11434/v1
#     LLM_API_KEY=ollama
#     LLM_MODEL=gemma3:4b

uvicorn app:app          # then open http://localhost:8000
```

`python extract.py` runs the extraction torture tests; `python test_guard.py`
verifies the deterministic guards; `python seed.py --count N --workers K`
regenerates the population.

## Stack

FastAPI · vanilla JS single-page UI · Gemma 3 4B on AMD Instinct MI300X via
vLLM/ROCm · sentence-transformers MiniLM (local CPU embeddings) · numpy ·
in-memory sessions (demo scope)

---

*Built by Ram — an introvert who couldn't initiate, building the app he needed.*
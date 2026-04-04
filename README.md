---
title: Disaster Relief Logistics
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<div align="center">

# 🚁 SupplyChain-Router
### *AI-Powered Disaster Relief Logistics*

**A hackathon submission for the [OpenEnv Challenge](https://github.com/raun/openenv-course) by Meta & Hugging Face**

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/electrifiedchan/disaster-relief-logistics)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-orange)](https://github.com/raun/openenv-course)
[![NVIDIA NIM](https://img.shields.io/badge/AI-NVIDIA%20NIM-green)](https://build.nvidia.com)

</div>

---

## 🌍 The Real Problem — In Plain English

Imagine a category-5 hurricane just hit a coastal city. Hospitals are flooded, roads are blocked, and the only way to get medicine, food, and emergency equipment to survivors is **by helicopter**.

A coordinator at a relief base has 6 pallets of supplies on the tarmac:

> *Pallet A: Blood plasma — critical, 30 lbs, MEDICAL*
> *Pallet B: Chemical disinfectant — critical, 20 lbs, CHEMICAL*
> *Pallet C: Water canisters — standard, 50 lbs, safe*

Three helicopters are fuelled and ready. Each has a weight limit. And here is the catch nobody thinks about until it is too late:

> **If you load chemical disinfectant and blood plasma into the same helicopter, aviation regulations require steel containment barriers — adding 50 lbs instantly, which overloads the helicopter and grounds it.**

The coordinator has minutes to decide. Get it wrong and the helicopter is grounded. Get it right and lives are saved.

**This project teaches an AI to make that decision — correctly, every time, across Easy, Medium, and Hard disaster scenarios.**

---

## 💡 Our Solution

We built a **reinforcement learning training environment** — think of it as a flight simulator, but for AI decision-making. The AI plays the logistics puzzle across three escalating difficulty levels, learning which routing strategies save lives and which ones ground helicopters.

The AI brain is **Mistral Devstral-2** (one of the world's most capable reasoning models, served via NVIDIA NIM) and the training scaffold uses the **OpenEnv framework** — a standardized toolkit by Meta and Hugging Face for building AI training environments.

### What makes this hard:

| Challenge | Why it matters |
|-----------|---------------|
| 🏋️ **Weight limits** | Cannot just throw everything into one helicopter |
| 🚨 **Critical priority** | Blood plasma must fly before water canisters |
| ☣️ **Hazmat separation** | Chemical + medical on same helicopter = instant overload |
| 🔁 **No looping** | Agent is penalized for repeating the same action |
| 📊 **Blended scoring** | 60% weight utilization + 40% critical priority routing |

---

## 🚨 The Dynamic Weight Trap

This is the feature that separates this from a simple box-stacking exercise.

```
HARD MODE — What a dumb AI does:

  Heli_A (cap: 100 lbs)
    ├── Pallet_1: Blood Plasma    30 lbs  [MEDICAL]
    └── Pallet_2: Disinfectant   40 lbs  [CHEMICAL]  ← BAD IDEA
        ↳ IATA hazmat rule triggered
        ↳ Steel containment barrier: +50 lbs added automatically
        ↳ Total load: 120 lbs > 100 lbs limit
        ↳ Helicopter GROUNDED.  Final score: 0.00 ❌

HARD MODE — What our AI does:

  Heli_A (cap: 100 lbs)             Heli_B (cap: 100 lbs)
    └── Pallet_1: Blood Plasma         ├── Pallet_2: Disinfectant
        30 lbs  [MEDICAL] ✅           │   40 lbs  [CHEMICAL] ✅
                                       └── Pallet_6: More Chemical
                                           20 lbs  [CHEMICAL] ✅
  Chemical and medical are isolated.
  No trap triggered.  Final score: 0.807 ✅
```

The trap is modeled on real **IATA Dangerous Goods Regulations** — the international rules governing what you can co-load on an aircraft. An AI that avoids this trap understands cargo composition, not just weight.

---

## 📊 Live Results on Hugging Face

| Difficulty | Steps | AI Score | Threshold | Verdict |
|------------|-------|----------|-----------|---------|
| **EASY** | 4 | **0.900** | ≥ 0.80 | ✅ PASS |
| **MEDIUM** | 5 | **0.910** | ≥ 0.80 | ✅ PASS |
| **HARD** | 6 | **0.807** | ≥ 0.80 | ✅ PASS |

> All 15 actions across 3 episodes were decided by Mistral Devstral-2 via NVIDIA NIM. Zero rule-based fallbacks.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  HACKATHON JUDGE BOT                      │
│     connects once → runs 3 episodes in one session       │
└──────────────────────┬───────────────────────────────────┘
                       │  reset() → step() × N → reset() …
                       ▼
┌──────────────────────────────────────────────────────────┐
│           ENVIRONMENT SERVER  (Hugging Face Spaces)       │
│                                                          │
│  SupplyChainEnv                  Scoring Engine          │
│  ├── Episode 1: EASY    ──▶   0.60 × utilization        │
│  ├── Episode 2: MEDIUM  ──▶ + 0.40 × priority routing   │
│  └── Episode 3: HARD    ──▶   threshold: ≥ 0.80         │
│      (Dynamic Weight Trap)                               │
│                                                          │
│  OR-Tools Oracle  ← mathematical upper bound            │
└──────────────────────┬───────────────────────────────────┘
                       │  observation + reward each step
                       ▼
┌──────────────────────────────────────────────────────────┐
│                    AI AGENT  (inference.py)               │
│                                                          │
│   Mistral Devstral-2-123B  via  NVIDIA NIM               │
│                                                          │
│   Reads:   helicopter capacities, pallet weights,        │
│            hazmat classes, priority levels               │
│   Decides: which pallet goes on which helicopter         │
│   Avoids:  chemical + medical on the same aircraft       │
└──────────────────────────────────────────────────────────┘
```

---

## �️ Project Structure

```
SupplyChain-Router/
├── 📄 models.py          ← Data contracts (Action, Observation, State)
├── 📡 client.py          ← WebSocket client for the environment
├── 🤖 inference.py       ← AI agent: 3-episode loop with Mistral
├── 🧪 test_3modes.py     ← Judge simulator: runs all 3 difficulties
├── 📋 openenv.yaml       ← Environment manifest (task definitions)
├── 🐳 Dockerfile         ← Production container for Hugging Face
├── 📦 requirements.txt   ← Python dependencies
└── server/
    ├── 🌐 app.py         ← FastAPI + WebSocket server (port 7860)
    ├── ⚙️  environment.py ← Scenarios, hazmat trap, blended scoring
    └── 🔬 oracle.py      ← OR-Tools SCIP solver (mathematical baseline)
```

---

## 🚀 Run It Yourself

### Option A — Live Demo (zero setup)
```
https://electrifiedchan-disaster-relief-logistics.hf.space/docs
```

### Option B — Local

```bash
# 1. Clone
git clone https://github.com/electrifiedchan/SupplyChain-Router.git
cd SupplyChain-Router
pip install -r requirements.txt

# 2. Start the server
uvicorn server.app:app --reload --port 7860

# 3. Run AI agent (NVIDIA NIM key required — free at build.nvidia.com)
export NVIDIA_API_KEY=nvapi-...
python inference.py

# 4. Simulate the judge evaluation (3 modes, hazmat-aware greedy)
python test_3modes.py
```

---

## � Key Engineering Decisions

| Decision | Reason |
|----------|--------|
| **WebSocket sessions** | HTTP endpoints are stateless — WS keeps episode alive across multiple steps |
| **Blended float score** | Partial credit for utilization AND priority routing — more honest than binary win/lose |
| **Hazmat as weight modifier** | Captures real IATA aviation physics; forces the AI to reason about cargo composition |
| **Soft repetition penalty (−0.5)** | Agent recovers from a loop instead of hard-failing the episode |
| **OR-Tools SCIP oracle** | Mathematical solver shows the theoretical optimum at episode end |
| **SUPPORTS_CONCURRENT_SESSIONS** | Multiple judges can evaluate simultaneously without state collision |
| **Episode counter fix** | `__init__` no longer consumes episode slot 1 — judge gets EASY → MEDIUM → HARD in correct order |

---

## 🏆 Why This Stands Out

Most logistics environments route packages to warehouses. This one routes **blood plasma and hazardous chemicals to disaster zones under real aviation safety constraints**.

The Dynamic Weight Trap is modeled on actual IATA Dangerous Goods Regulations. An AI that navigates it correctly is not playing a game — it is demonstrating the same reasoning a trained logistics officer uses when loading a medevac helicopter.

---

## 📚 Built With

| Tool | Role |
|------|------|
| [OpenEnv](https://github.com/raun/openenv-course) | Environment framework (Meta × Hugging Face) |
| [Mistral Devstral-2-123B](https://build.nvidia.com/mistralai/devstral-2) | AI reasoning via NVIDIA NIM |
| [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) | Async WebSocket server |
| [OR-Tools SCIP](https://developers.google.com/optimization) | Mathematical optimization oracle |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Production deployment (Docker, port 7860) |
| [Pydantic v2](https://docs.pydantic.dev) | Type-safe data models |

---

<div align="center">

*Built for the OpenEnv Hackathon — April 2026*

</div>

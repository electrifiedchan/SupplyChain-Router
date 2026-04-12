# SupplyChain-Router
*Expert AI-driven disaster relief logistics routing using OpenEnv.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-orange.svg)
![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-purple.svg)

**_SupplyChain-Router is an autonomous multi-constraint logistics pipeline that stress-tests LLM agent reasoning against dynamic hazard physics and stochastic weather anomalies._**

## Introduction 🚁

SupplyChain-Router abandons traditional, static bin-packing problems by forcing an inference agent into a dynamically collapsing ruleset. Standard greedy heuristics will violently fail here. Built for hackathon evaluators and researchers testing emergent reasoning, this environment forces the LLM to actively interpret live METAR telemetry, juggle irreversible containment penalties, and optimize routing mathematically while its available capacity literally vanishes out from under it.

## How Scoring Works 🎯

The routing solution's quality is aggressively graded against a tightly calculated capacity baseline. It relies on a weighted sum of payload utilization and critical item delivery rate. 

Formula:
```python
score = 0.60 * (useful_weight / baseline_capacity) + 0.40 * (routed_criticals / total_criticals)
```

The evaluator relies on a strictly exact log regex. An output like the following must be safely parsed without standard framework noise bleeding in:
```bash
[END] task=disaster-relief-hard (episode 3/3) score=0.815 steps=6
```

> **Why this matters:** A score of exactly `0.815` represents the absolute maximum mathematical ceiling in Hard Mode, not `1.0`. The Hard payload totals exactly 180 lbs against a `260` lb baseline (`180/260 = 0.692`). The reward curve specifically avoids `1.0` to force algorithms to fight for fractional optimizations over baseline capacity rather than capping out early on superficial completion.

## Mode Comparison 📊

<div align="center">

| Mode | Helicopters | Total Capacity | Pallets | Total Payload | Active Hazards | Score Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🟢 Easy | `Heli_A`, `Heli_B` | 120 lbs (`60/60`) | 4 | 100 lbs | None | ~0.900 |
| 🟡 Medium | `Heli_A`, `Heli_B`, `Heli_C` | 220 lbs (`80/80/60`) | 6 | 180 lbs | None | ~0.891 |
| 🔴 Hard | `Heli_A`, `Heli_B`, `Heli_C` | 260 lbs (`110/110/40`) | 6 | 180 lbs | Tornado & Hazmat | ~0.815 |

</div>

## Hard Mode Deep Dive ⚡

This is the showcase execution level. Every rule is weaponised to punish careless LLM generations. 

### The METAR Injection
At Step 3, the telemetry throws an aviation weather special: `SPECI KTBW 141347Z 18032G58KT 1/4SM +TSRA FG BKN005 OVC010CB`.
In plain English: A severe thunderstorm with heavy rain (`+TSRA`), fog (`FG`), winds gusting to 58 knots (`18032G58KT`), and quarter-mile visibility (`1/4SM`) is approaching `Heli_C`. The LLM must immediately abandon routing to this vehicle, because in exactly one step, it will be grounded and all onboard cargo will be destroyed.

### The Containment Trap
Mixing `medical` and `chemical` pallets on the same helicopter triggers the absolute destruction of a perfectly calculated manifest. If `Heli_A` has biological medical supplies and receives an incoming battery chemical pallet, the environment immediately generates a non-reversible +50 lb containment penalty representing emergency protective equipment. In a scenario where total tight packing requires exactly 180 out of 260 available lbs, a sudden 50 lb balloon ensures a physics violation (capacity overflow) and collapses the score to `0.001`.

### The Pydantic Grounding Sequence
> **Why this matters:** A helicopter cannot have a maximum capacity lower than its current load without invalidating its own physical existence in memory. 

When the tornado grounds `Heli_C`, its capacity becomes `0`. However, because the server utilizes strict declarative Pydantic `validate_assignment` models, setting capacity to `0` while cargo is loaded instantly throws a `ValueError` inside the environment logic. Thus, the mutation step carefully zeroes `current_load`, clears the `loaded_pallets` array, and *then* zeroes `max_capacity`.

### The [END] Regex Spec
Hackathon evaluators use dumb pipeline scrapers. The regex engine looking for the final outcome string runs strictly to the character format. A careless trailing space, a newline embedded in a dictionary string, or framework logging overlap will result in a blind parse error, falsely zeroing out a perfectly valid run in the autonomous grader.

## Quickstart 🚀

```bash
git clone https://github.com/electrifiedchan/SupplyChain-Router.git
cd SupplyChain-Router
uv pip install -r requirements.txt
uvicorn server.environment:app --host 0.0.0.0 --port 8000
python inference.py
```

## Repository Structure 📂

- `server/environment.py`: The OpenEnv simulation engine, maintaining physics bounds, telemetry injections, and evaluation logic.
- `models.py`: Immutable state rules enforcing type invariants, maximum capacity ceilings, and array hygiene before actions reach the firewall.
- `inference.py`: Agent execution pipeline connecting NVIDIA NIM endpoints to the game state, providing prompt generation and regex-safe termination.
- `client.py`: The asynchronous WebSocket layer binding the local agent to the OpenEnv remote simulator cleanly.
- `requirements.txt`: Pinpointing essential framework dependencies including OpenEnv core and fastmcp.

## Design Decisions 📐

The weight ratio in Medium Mode wasn't accidental. By scaling six pallets to 180 lbs against exactly 220 lbs of baseline vehicle availability, standard First-Fit Decreasing heuristic bins will inevitably fragment contiguous space. Placements that look "good enough" early on will force the agent to orphan highly critical pallets at the end when exactly zero 35 lb gaps remain. The LLM cannot simply react; it must branch and bound the solution fully before its first real turn.

Delaying the Tornado anomaly until Step 3 and 4 builds an irresistible trap. Immediate knowledge of a broken helicopter merely turns it into a smaller bin-packing puzzle. Holding the warning forces the model into temporal state changes. It might have already packed 20 lbs of cargo into `Heli_C` before the `SPECI` drops, only to suddenly realise that it must redirect its entire forward strategy while absorbing the loss. This actively tests multi-step re-planning rather than static upfront calculation.

The Regex logger lockdown ensures that any evaluation environment operates objectively on pure, immutable numbers regardless of how noisy the underlying inference engine gets. Models hallucinate, libraries spam standard out, and exception handlers throw tracebacks. The final payload syntax is essentially an airlock, ensuring that the graded score is the absolute only truth that reaches the upstream judging infrastructure, shielding against exploits and parse failures alike.

Disabling concurrent sessions prevents the underlying singleton physics engine from bleeding temporal mutations across evaluator calls. If Thread A grounds `Heli_C` in Step 4, and Thread B arrives asking for Step 1 logic, the engine state becomes a fundamentally unreliable race condition. This forces serial stability out of the box, trading high-frequency throughput for deterministic, hackathon-ready precision scoring.

## License 📄

MIT License. Built for the OpenEnv Multi-Agent logistics framework showcase.

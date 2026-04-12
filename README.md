# SupplyChain-Router
> Expert AI-driven disaster relief logistics routing using OpenEnv.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-orange)

## Hero Section
SupplyChain-Router is an autonomous logistics AI pipeline designed to solve multi-constraint disaster relief routing. It elevates standard bin-packing by introducing dynamic hazards, strict segregation penalties, and stochastic weather anomalies that actively punish greedy heuristics. Built for hackathon evaluators, researchers, and LLM reasoning judges, it rigorously stress-tests an agent's ability to interpret live telemetry and navigate complex, mutating rulesets.

## Architecture Diagram

| Mode | Capacity Limit (lbs) | Dynamic Weight Trap | Telemetry Anomaly | Max Score Ceiling |
|------|----------------------|----------------------|--------------------|-------------------|
| **Easy** | 120 (60+60) | ❌ Inactive | ❌ None | ~1.000 |
| **Medium**| 220 (80+80+60) | ❌ Inactive | ❌ None | ~1.000 |
| **Hard** | 260 (110+110+40) | ✅ +50 lb Containment | ✅ Heli_C Grounded at Step 4 | ~0.815 |

```
Final Score = (0.60 × Utilization) + (0.40 × Criticals)
```

## How It Works

### Easy Mode
The agent is introduced to a basic bin-packing task. The payload consists of 4 standard/safe pallets weighing a total of 100 lbs loaded onto two helicopters (`Heli_A` and `Heli_B`), each with a 60 lb capacity. The agent must successfully route the pallets without overflowing any single vehicle's limit. No hazards or dynamically evolving constraints apply.

### Medium Mode
Tightened knapsack math becomes the core challenge. The environment introduces `Heli_C` (60 lb capacity) alongside `Heli_A` (80 lb) and `Heli_B` (80 lb). The payload scales to 6 pallets totaling 185 lbs, but the baseline capacity strictly allocates 220 lbs. While hazards exist, the containment penalty is not applied. The LLM must distribute loads methodically without overflowing.

### Hard Mode
The environment introduces asymmetric capacity rules, extreme weather, and hazmat physics. Helicopter capacities are uneven: `Heli_A` (110 lb), `Heli_B` (110 lb), `Heli_C` (40 lb).
- **METAR/NOTAM Injection**: At Step 3, the agent receives a `CRITICAL TELEMETRY WARNING` that a severe weather system is approaching `Heli_C`.
- **Heli_C Grounding**: By Step 4, `Heli_C` is permanently grounded (capacity becomes 0). Any cargo placed there beforehand is destroyed.
- **Dynamic Weight Trap**: Mixing `medical` and `chemical` pallets on the same helicopter triggers a permanent +50 lb containment penalty to that vehicle. The LLM must proactively segregate hazards to avoid a catastrophic overflow.

## Scoring System

The routing solution's quality is graded on a weighted sum of payload utilization and critical delivery rate:

**Formula:**
`Final score = 0.60 × (useful weight routed / baseline capacity) + 0.40 × (critical pallets routed / total criticals)`

A perfect Hard Mode routing of all 6 pallets (180 lb out of 260 lb baseline) and all 3 criticals equates to `0.60 × (180/260) + 0.40 × (3/3) ≈ 0.815`. Triggering the containment trap, overflowing a helicopter, or leaving critical pallets unloaded significantly lowers the score toward a defined failure floor of `0.001`.

**Grading Format:**
The openenv auto-grader evaluates performance strictly through standard regex constraints for logging final execution:
```
[END] task=disaster-relief-hard (episode 3/3) score=0.815 steps=6
```

## Quickstart

```bash
# Clone the repository
git clone https://github.com/electrifiedchan/SupplyChain-Router.git
cd SupplyChain-Router

# Install requirements
uv pip install -r requirements.txt

# Start the environment server locally
uvicorn server.environment:app --host 0.0.0.0 --port 8000

# Run the inference pipeline
python inference.py
```

Expected Output:
```
[START] task=disaster-relief-easy env=OpenEnv-SupplyChain model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"helicopter_id": "Heli_A", "pallet_id": "Pallet_3"} reward=0.67 done=false error=null
...
[END] task=disaster-relief-easy (episode 1/3) score=0.985 steps=4
```

## Repository Structure

- `server/environment.py`: The core OpenEnv server defining capacity constraints, hazards, and tornado telemetry events.
- `models.py`: Pydantic definitions enforcing strict datatype and capacity validations before actions reach the firewall.
- `inference.py`: The main solver coordinating evaluation, greedy fallbacks, logging parsers, and prompt configurations.
- `requirements.txt`: Identifies core framework dependencies including OpenEnv, Pydantic, and fastmcp.
- `client.py`: Interfacing capabilities for WebSocket and EnvServer communications.

## Design Decisions

**Why 1:1 weight ratio in Medium Mode**
Medium Mode is meticulously crafted to leave zero wiggle room in pure First-Fit heuristics. By aligning total payload and total capacities to narrow margins, the LLM is forced to consider granular combinations instead of greedy filling. Sub-optimal piece placement inevitably orphans the critical remaining pallets.

**Why loaded_pallets must be cleared before max_capacity is zeroed**
Because Pydantic `validate_assignment` natively enforces strict model states, setting capacity to 0 while a 40 lb load sits onboard immediately raises a model-layer ValueError. Ordering the sequence—clear capacity *then* zero limit—prevents internal server cascades during Hard Mode Step 4 tornado injections.

**Why [END] format is locked to regex scraper spec**
Execution validation relies upon automated scraping by upstream evaluators running non-interactive environments. The `[END] task=X score=Y steps=Z` log strictly decouples arbitrary framework verbosity from final submission integrity, sealing vulnerabilities like CVE-4.1.

**Why anomaly fires at step 3/4 specifically**
Delaying the Tornado anomaly injection to Steps 3 and 4 forces branching. Giving early notice prevents unpreventable failure, but holding the actual payload destruction until Step 4 forces the LLM to comprehend temporal state—changing plans mid-operation, rather than reacting passively to static conditions.

## Known Constraints & Edge Cases

- **Greedy Fallback**: When LLM calls consistently fail syntax parsing or violate physics bounds beyond the retry limits, `inference.py` relies on a Hazmat-aware First-Fit-Decreasing routine. This masks absolute agent failure but yields heavily penalized scores.
- **SUPPORTS_CONCURRENT_SESSIONS = False**: Because telemetry mutations fundamentally mutate universal scenario blueprints (like permanently grounding Helicopter C), parallel instance invocations would trigger state-contamination. Requests are handled sequentially.

## License & Acknowledgements

MIT License. Built for the OpenEnv Multi-Agent framework showcase.

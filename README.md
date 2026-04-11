---
title: Disaster Relief Logistics
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🚁 AI-Driven Disaster Logistics: SDVRPTW Hyper-Heuristic Solver

## ⚠️ The Emotional Hook & Zero Hour
According to the World Health Organization (WHO), 50% of vaccines are wasted annually due to routine cold chain failures. In a disaster zone, this spoilage window compresses exponentially. Blood products expire in 4-6 hours without refrigeration; insulin denatures rapidly above 37°C. Every routed decision is a hard time-window constraint where failure results in critical payload loss.

## 🚧 The Friction: The Collapse of Static Routing
During the 2017 Hurricane trio (Harvey, Irma, Maria) and the 2015 Nepal Earthquake, traditional centralized dispatching and deterministic routing algorithms catastrophically failed. Standard APIs assume static graphs. When physical infrastructure is destroyed, static routing requires manual whiteboard calculation, creating massive cognitive overload for dispatchers. Recent operations research proves that exact methods for Dynamic Vehicle Routing Problems with Time Windows (DVRPTW) are computationally infeasible in highly stochastic environments.

## ✨ The Magic: LLM-Augmented Semantic Routing
This environment establishes Large Language Models as superior hyper-heuristic solvers for stochastic logistics. Instead of relying on rigid JSON APIs, this AI agent directly parses unstructured, real-time machine telemetry and navigates complex combinatorial constraints.

### ⚙️ Core Technical Differentiators:
1. **Asymmetric Combinatorial Defense:** Defeats basic First-Fit Decreasing (FFD) heuristics by enforcing mathematically rigorous subset-sum branching. Total payload strictly equals fleet capacity (180 lbs) across asymmetrical nodes.
2. **Two-Step Semantic Parsing:** Injects raw aviation telemetry (SPECI/METAR strings: `18032G58KT 1/4SM +TSRA FG BKN005 OVC010CB` and NOTAMs) mid-flight. Decouples text warnings from physical constraints, mathematically proving the LLM is executing semantic reading comprehension, not just reacting to JSON state changes.
3. **Execution Firewall:** Implements secure `try...except` interception of hallucinated entity IDs, ensuring environment stability and valid penalty application without fatal `KeyError` crashes.
4. **Persistent Telemetry Memory:** Eliminates LLM attention drift by pinning critical dynamic constraints directly into the active observation space.

## 🚀 Execution Instructions
Run the pre-configured inference loop to observe the LLM navigate the stochastic METAR weather anomaly:
```bash
python inference.py
```

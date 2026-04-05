"""
3-Mode Judge Evaluation Test
Runs EASY → MEDIUM → HARD on the live HF Space using hazmat-aware greedy.
"""
import asyncio
import os
from client import SupplyChainEnvClient
from models import LogisticsAction

API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("NVIDIA_API_KEY") or "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"
ENV_URL = "https://electrifiedchan-disaster-relief-logistics.hf.space"
SUCCESS_THRESHOLD = 0.80


def hazmat_aware_greedy(obs: dict, full_pallet_ref: dict):
    """Capacity + hazmat-aware greedy. Avoids putting chemical+medical on same heli."""
    pallets = obs.get("remaining_pallets", {})
    helis = obs.get("helicopters", {})

    # Determine which hazard classes are already on each helicopter
    heli_hazards: dict = {}
    for hid, heli in helis.items():
        loaded = heli.get("loaded_pallets", [])
        hazards = set()
        for pid in loaded:
            if pid in full_pallet_ref:
                h = full_pallet_ref[pid].get("hazard_class", "safe")
                if h != "safe":
                    hazards.add(h)
        heli_hazards[hid] = hazards

    # Sort pallets: critical first, then heavier first
    def pallet_sort_key(kv):
        pid, p = kv
        prio = 0 if (p.get("priority") if isinstance(p, dict) else getattr(p, "priority", "standard")) == "critical" else 1
        w = (p.get("weight") if isinstance(p, dict) else getattr(p, "weight", 0))
        return (prio, -w)

    for pid, pallet in sorted(pallets.items(), key=pallet_sort_key):
        ph = (pallet.get("hazard_class") if isinstance(pallet, dict) else getattr(pallet, "hazard_class", "safe")) or "safe"
        pw = (pallet.get("weight") if isinstance(pallet, dict) else getattr(pallet, "weight", 0)) or 0

        # Sort helis by most remaining capacity
        for hid, heli in sorted(helis.items(),
                                 key=lambda kv: -(kv[1].get("max_capacity", 0) - kv[1].get("current_load", 0))):
            room = heli.get("max_capacity", 0) - heli.get("current_load", 0)
            if room < pw:
                continue
            if ph != "safe":
                existing = heli_hazards.get(hid, set())
                if (ph == "chemical" and "medical" in existing) or (ph == "medical" and "chemical" in existing):
                    continue
            return hid, pid

    # Absolute fallback
    for pid in pallets:
        for hid in helis:
            return hid, pid
    return None, None


async def run_episode(env: SupplyChainEnvClient, episode_num: int):
    result = await env.reset()
    obs = result.observation.model_dump()
    diff = obs.get("task_difficulty", "?")
    print(f"\n=== EPISODE {episode_num} | DIFFICULTY: {diff.upper()} ===")

    # Capture full pallet reference BEFORE any are removed
    full_pallet_ref: dict = {}
    for pid, p in obs.get("remaining_pallets", {}).items():
        full_pallet_ref[pid] = p if isinstance(p, dict) else p.model_dump()

    rewards = []
    score = 0.0
    for step in range(1, 20):
        if obs.get("done", False):
            break
        if not obs.get("remaining_pallets") or not obs.get("helicopters"):
            break

        hid, pid = hazmat_aware_greedy(obs, full_pallet_ref)
        if not hid or not pid:
            print("  No valid action found!")
            break

        r = await env.step(LogisticsAction(helicopter_id=hid, pallet_id=pid))
        obs = r.observation.model_dump()
        reward = float(r.reward or 0.0)
        rewards.append(reward)

        info = obs.get("info", {})
        reason = (info.get("reason", "") if isinstance(info, dict) else "")[:70]
        trap = info.get("dynamic_weight_trap_triggered", False) if isinstance(info, dict) else False
        trap_str = "  ⚠ TRAP!" if trap else ""
        print(f"  Step {step}: {hid} <- {pid} | reward={reward:.3f} done={r.done}{trap_str}")
        print(f"    {reason}")
        if r.done:
            score = reward
            break

    ok = score >= SUCCESS_THRESHOLD
    result_label = "PASS" if ok else "FAIL"
    print(f"  --> score={score:.3f}  [{result_label}]  rewards={rewards}")
    return ok, score, diff


async def main():
    all_results = []
    async with SupplyChainEnvClient(base_url=ENV_URL) as env:
        for ep in range(1, 4):
            ok, score, diff = await run_episode(env, ep)
            all_results.append((diff, score, ok))

    print("\n" + "=" * 55)
    print("JUDGE-STYLE 3-MODE EVALUATION SUMMARY")
    print("=" * 55)
    for diff, score, ok in all_results:
        label = "PASS" if ok else "FAIL"
        print(f"  {diff.upper():8s}  score={score:.3f}  [{label}]")
    passed = sum(1 for _, _, ok in all_results if ok)
    print(f"\n  Total: {passed}/3 modes passed (threshold >= {SUCCESS_THRESHOLD})")
    if passed == 3:
        print("  ALL MODES PASS - ready for judge evaluation!")
    else:
        print("  Some modes need attention.")


if __name__ == "__main__":
    asyncio.run(main())

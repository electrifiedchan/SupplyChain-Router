"""
Random Action Baseline
Sends uniformly random (helicopter_id, pallet_id) pairs each step.
Runs 9 episodes (3 sessions × EASY/MEDIUM/HARD) against the live HF Space.
Each session gets a fresh WebSocket connection so a SERVER_ERROR on one
episode doesn't corrupt later episodes.
"""
import asyncio
import random
from client import SupplyChainEnvClient
from models import LogisticsAction

ENV_URL = "https://electrifiedchan-disaster-relief-logistics.hf.space"
LLM_AVG = 0.872


async def run_one_episode(env: SupplyChainEnvClient, ep: int) -> tuple[float, str]:
    """Returns (score, difficulty). Score is 0.0 on any error or crash."""
    try:
        result = await env.reset()
        obs = result.observation.model_dump()
        diff = obs.get("task_difficulty", "?")
        score = 0.0

        for _ in range(15):
            if obs.get("done", False):
                break
            pallets = list((obs.get("remaining_pallets") or {}).keys())
            helis = list((obs.get("helicopters") or {}).keys())
            if not pallets or not helis:
                break
            hid = random.choice(helis)
            pid = random.choice(pallets)
            try:
                r = await env.step(LogisticsAction(helicopter_id=hid, pallet_id=pid))
            except RuntimeError as e:
                print(f"    [RuntimeError on step] {e}")
                return 0.0, diff
            obs = r.observation.model_dump()
            if r.done:
                score = float(r.reward or 0.0)
                break

        print(f"  Ep{ep} [{diff:6s}]: score={score:.3f}")
        return score, diff

    except Exception as e:
        print(f"  Ep{ep} [ERROR]: {e}")
        return 0.0, "unknown"


async def main() -> None:
    all_results: list[tuple[str, float]] = []

    # 3 sessions × 3 episodes each (EASY → MEDIUM → HARD per session)
    # If a SESSION_ERROR occurs mid-session, that episode scores 0.0 and
    # we open a fresh connection for the next session.
    for session in range(1, 4):
        print(f"\n--- Session {session} (new connection) ---")
        try:
            async with SupplyChainEnvClient(base_url=ENV_URL) as env:
                for ep in range(1, 4):
                    score, diff = await run_one_episode(env, ep)
                    all_results.append((diff, score))
        except Exception as e:
            print(f"  Session {session} connection failed: {e}")
            # pad missing episodes with 0.0 so totals stay correct
            while len(all_results) < session * 3:
                all_results.append(("unknown", 0.0))

    print(f"\n{'='*55}")
    print(f"RANDOM BASELINE — {len(all_results)} episodes")
    print(f"{'='*55}")
    for diff, score in all_results:
        print(f"  {diff:8s}  {score:.3f}")

    avg = sum(s for _, s in all_results) / len(all_results)
    zeros = sum(1 for _, s in all_results if s == 0.0)
    print(f"\n  Random avg score  : {avg:.3f}")
    print(f"  LLM avg score     : {LLM_AVG}")
    print(f"  Zero-score (crash): {zeros}/{len(all_results)}")
    if avg > 0:
        print(f"  Dynamic range     : {LLM_AVG / avg:.1f}x over random")
    else:
        print(f"  Dynamic range     : ∞  (random agent crashed every episode)")


if __name__ == "__main__":
    asyncio.run(main())

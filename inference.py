import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI
from client import SupplyChainEnvClient
from models import LogisticsAction

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 1. MANDATORY LOGGING FORMATTERS ─────────────────────────────────────────

def safe_score(raw_score: float) -> float:
    """Guarantees score is strictly between 0 and 1"""
    return max(0.01, min(0.99, float(raw_score)))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(
    step: int,
    action: dict,
    reward: float,
    done: bool,
    error: str = None,
) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    import json
    action_str = json.dumps(action) if action else "{}"
    
    safe_reward = safe_score(reward)
    
    print(
        f"[STEP] step={step} action={action_str} reward={safe_reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )

def log_end(task_id: str, score_val: float, steps: int) -> None:
    final_score = safe_score(score_val)
    print(f"[END] task={task_id} score={final_score:.2f} steps={steps}", flush=True)

# ─── 2. SETTINGS & AUTH (Validator Compliant) ───────────────────────────────

import sys
import os
from openai import OpenAI

# The validator explicitly injects API_BASE_URL and API_KEY.
# We MUST use these exactly as named.
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

# Safety Check: If the validator didn't provide them, fallback to defaults 
# ONLY for local testing. In the portal, these will be set.
if not API_BASE_URL:
    API_BASE_URL = "https://router.huggingface.co/hf-inference/v1"
if not API_KEY:
    API_KEY = "NO_KEY_SET"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://electrifiedchan-disaster-relief-logistics.hf.space")
MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.80  # blended score must hit 0.80 to count as success
MAX_RETRIES = 2
RETRY_DELAY = 2.0

# ─── 3. PROMPT BUILDER ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert disaster-relief logistics AI. Your job is to route cargo pallets
onto helicopters one at a time. Each action you output must be a single JSON object.

══════════════════════════════════════════════════════════
OUTPUT FORMAT — NON-NEGOTIABLE
══════════════════════════════════════════════════════════
Every response must be exactly this JSON structure and nothing else:

{
  "hazard_check": "<your one-sentence reasoning about hazard compatibility>",
  "helicopter_id": "<Heli_A | Heli_B | Heli_C>",
  "pallet_id": "<Pallet_1 through Pallet_6>"
}

The "hazard_check" field is your private scratchpad. It is ignored by the
environment but forces you to verify safety before committing to an action.
If a list of PRE-VERIFIED LEGAL MOVES is provided, you must select your action exclusively from that list.
Do not output any text outside this JSON object.

══════════════════════════════════════════════════════════
HARD RULE — CARGO SEGREGATION (READ THIS BEFORE EVERY MOVE)
══════════════════════════════════════════════════════════
MEDICAL and CHEMICAL pallets must NEVER share the same helicopter.

If you load a CHEMICAL pallet onto a helicopter that already carries any
MEDICAL pallet — or vice versa — the environment immediately applies a
+50 lb CONTAINMENT PENALTY to that helicopter's current load.

This penalty is PERMANENT and CANNOT be undone.
In most cases it will cause the next pallet to overflow the capacity limit,
which triggers a Physics Violation and ends the episode with score = 0.001.

SAFE pallets have no restriction. They can fly with any cargo.

══════════════════════════════════════════════════════════
EXAMPLE — WHAT NOT TO DO (study this carefully)
══════════════════════════════════════════════════════════
Situation:
  Heli_A manifest: [Pallet_1 (MEDICAL, 40 lb)]  — load = 40/110 lb
  Remaining: Pallet_2 (CHEMICAL, 40 lb)

WRONG action:
  {"hazard_check": "Heli_A has space, loading Pallet_2.", "helicopter_id": "Heli_A", "pallet_id": "Pallet_2"}

What happens:
  MEDICAL + CHEMICAL on same helicopter → TRAP fires.
  Heli_A load becomes 40 (existing) + 50 (penalty) = 90 lb.
  Then Pallet_2 (40 lb) is added: 90 + 40 = 130 lb > 110 lb capacity.
  Result: Physics Violation. Episode ends. Score = 0.001.

CORRECT action:
  {"hazard_check": "Heli_A has MEDICAL cargo. Pallet_2 is CHEMICAL. Incompatible. Route to Heli_B which has no medical.", "helicopter_id": "Heli_B", "pallet_id": "Pallet_2"}

Why it works:
  Heli_B has no medical cargo, so chemical is safe there.
  No penalty. No overflow. Mission continues.

══════════════════════════════════════════════════════════
STRATEGY — HOW TO ROUTE CORRECTLY EVERY TIME
══════════════════════════════════════════════════════════
Step 1 — Sort pallets into three groups before your first move:
  • MEDICAL group:   Pallet_1, Pallet_5
  • CHEMICAL group:  Pallet_2, Pallet_6
  • SAFE group:      Pallet_3, Pallet_4
  *CRITICAL RULE: Pallet weights change depending on the difficulty mode. NEVER assume a pallet's weight from past examples. ALWAYS read the live observation state to verify the exact weight before calculating free capacity.*

Step 2 — Assign one helicopter to MEDICAL, one to CHEMICAL:
  • Pick one helicopter (e.g. Heli_A) → receives ALL medical pallets.
  • Pick another helicopter (e.g. Heli_B) → receives ALL chemical pallets.
  • NEVER mix these two helicopters' assigned hazard class.
  • SAFE pallets fill remaining space on either helicopter.

Step 3 — Handle the TORNADO anomaly (Hard Mode only):
  • In Hard Mode, Heli_C will be struck by a tornado and grounded at Step 4.
  • ANY cargo placed on Heli_C before the tornado hits will be DESTROYED, instantly ruining your score.
  • Therefore, NEVER route any pallets to Heli_C in Hard Mode. Ignore it completely from Step 1. Use only Heli_A and Heli_B.
  • (In Medium Mode, there is no tornado, and you MUST use Heli_C to fit all 180 lb of cargo).

Step 4 — Prioritise CRITICAL pallets:
  • Pallet_1, Pallet_2, Pallet_3 are marked critical.
  • Route critical pallets before standard ones where possible.
  • The scoring formula rewards critical delivery at 40% weight.
  • PRO TIP: Always load Pallet_5 and Pallet_6 BEFORE Pallet_4 to ensure you do not run out of hazardous capacity on your helicopters.

══════════════════════════════════════════════════════════
SCORING (so you understand what you are optimising)
══════════════════════════════════════════════════════════
  Final score = 0.60 × (useful weight routed / 260 lb baseline)
              + 0.40 × (critical pallets routed / 3 total criticals)

  Perfect routing (all 6 pallets, all 3 criticals, no trap) → score ≈ 0.815
  Any trap trigger or unrouted pallet reduces score toward 0.001.

══════════════════════════════════════════════════════════
CHECKLIST — RUN THIS BEFORE EVERY ACTION
══════════════════════════════════════════════════════════
Before choosing helicopter_id and pallet_id, answer in hazard_check:

  1. CAPACITY MATH: What is the exact weight of the pallet, and what is the free capacity of the target helicopter? Does it fit? (If no, pick a different helicopter).
  2. LIVE HAZARD TRUTH: Look directly at the LIVE ENVIRONMENT STATE right now. What is the EXACT hazard_class value for this pallet? (Do not guess or assume!).
  3. What hazard classes are already loaded on my target helicopter?
  4. Is there a MEDICAL+CHEMICAL conflict? If yes, pick a different helicopter.
  5. Have I received a METAR/NOTAM warning? If yes, avoid Heli_C immediately.
  6. Is Heli_C already grounded (capacity=0)? If yes, never route to it.
"""
def _is_small_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    for marker in ["70b", "72b", "123b", "devstral", "mixtral"]:
        if marker in name_lower:
            return False
    return True

def _build_legal_moves_block(obs: Dict[str, Any]) -> str:
    _detect_grounded_heli(obs)
    pallets = obs.get("remaining_pallets", {})
    helicopters = obs.get("helicopters", {})
    difficulty = obs.get("task_difficulty", "unknown")
    
    HAZARD_PRIORITY = {"medical": 0, "chemical": 1, "safe": 2}
    sorted_pallets = sorted(
        pallets.items(),
        key=lambda x: (
            HAZARD_PRIORITY.get(x[1].get("hazard_class", "safe"), 2),
            -x[1].get("weight", 0),
        )
    )

    moves = []
    for p_id, p_data in sorted_pallets:
        weight = p_data.get("weight", 0)
        hazard = p_data.get("hazard_class", "safe")
        priority_str = f" [CRITICAL]" if p_data.get("priority") == "critical" else ""
        hazard_str = f"{hazard.upper()}{priority_str}"
        
        valid_helis = {
            h_id: h_data for h_id, h_data in helicopters.items()
            if not (_grounded_heli_id is not None and h_id == _grounded_heli_id)
        }
        
        for h_id, h_data in sorted(valid_helis.items(), key=lambda x: (x[1]["max_capacity"] - x[1]["current_load"])):
            free = h_data["max_capacity"] - h_data["current_load"]
            if weight > free:
                continue
            if _would_trigger_trap(h_id, hazard):
                continue
            
            moves.append(
                f'  {{"helicopter_id": "{h_id}", "pallet_id": "{p_id}"}} — {p_id} ({hazard_str}, {weight} lb) → {h_id} ({free} lb free)'
            )
            
    if not moves:
        return ""
        
    block = "\n✅ PRE-VERIFIED LEGAL MOVES\n══════════════════════════════════\n"
    block += "Every move below has already passed capacity and hazmat checks.\n"
    block += "Any move NOT on this list will be rejected by the environment.\n\n"
    block += "\n".join(moves)
    block += "\n\nApply your strategy (hazard segregation, critical-first, tornado avoidance)\n"
    block += "to pick the BEST move from the list above.\n"
    return block

def _build_prompt(obs: Dict[str, Any]) -> str:
    """Full detailed prompt including hazmat trap rules and live telemetry."""
    pallets = obs.get("remaining_pallets", {})
    helicopters = obs.get("helicopters", {})
    difficulty = obs.get("task_difficulty", "unknown")
    step = obs.get("step_count", "?")
    info = obs.get("info", {})
    alert_msg = info.get("reason", "")
    
    telemetry_section = f"\n🚨 LIVE SYSTEM ALERT:\n{alert_msg}\n" if alert_msg and alert_msg != "Action accepted." else ""

    heli_lines = []
    for h_id, h in helicopters.items():
        free = h["max_capacity"] - h["current_load"]
        trap = h.get("containment_penalty_active", False)
        loaded = h.get("loaded_pallets", [])
        trap_warning = " ⚠️ HAZMAT MIXED - DO NOT ADD chemical/medical" if trap else ""
        heli_lines.append(
            f"  {h_id}: {h['current_load']}/{h['max_capacity']} lb used | "
            f"{free} lb free | loaded={loaded} | "
            f"containment_penalty_active={trap}{trap_warning}"
        )

    base_prompt = f"""{SYSTEM_PROMPT}

══════════════════════════════════════════════════════════
LIVE ENVIRONMENT STATE
══════════════════════════════════════════════════════════
Difficulty: {difficulty.upper()} | Step: {step}
{telemetry_section}
HELICOPTERS:
{chr(10).join(heli_lines)}

REMAINING PALLETS:
{json.dumps(pallets, indent=2)}
"""
    return base_prompt + _build_legal_moves_block(obs)

# ─── 4. ACTION VALIDATOR ──────────────────────────────────────────────────────

def _validate_cove_quality(action: Dict[str, Any], obs: Dict[str, Any]) -> Tuple[bool, str]:
    if "hazard_check" not in action:
        return False, "hazard_check field absent"
    
    hc = action["hazard_check"]
    if not isinstance(hc, str) or len(hc.strip()) < 20:
        return False, "hazard_check too short"

    p_id = action.get("pallet_id")
    pallet = obs.get("remaining_pallets", {}).get(p_id, {})

    hazard_class = pallet.get("hazard_class", "safe").lower()
    p_id_lower = str(p_id).lower() if p_id else ""

    hc_lower = hc.lower()
    if hazard_class not in hc_lower and p_id_lower not in hc_lower:
        return False, "Grounding rule violated: hazard class or pallet ID not quoted"

    return True, "CoVe valid"

def _validate_action(
    action: Dict[str, Any], obs: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check action IDs exist and weight fits before sending to environment."""
    h_id = action.get("helicopter_id")
    p_id = action.get("pallet_id")
    helicopters = obs.get("helicopters", {})
    pallets = obs.get("remaining_pallets", {})

    if not h_id or not isinstance(h_id, str):
        return False, f"Invalid helicopter_id: {h_id!r}"
    if not p_id or not isinstance(p_id, str):
        return False, f"Invalid pallet_id: {p_id!r}"
    if h_id not in helicopters:
        return False, f"Helicopter '{h_id}' not in {list(helicopters.keys())}"
    if p_id not in pallets:
        return False, f"Pallet '{p_id}' not in {list(pallets.keys())}"

    heli = helicopters[h_id]
    pallet = pallets[p_id]
    free = heli["max_capacity"] - heli["current_load"]

    # Check hazmat trap (would add +50 lb containment penalty)
    difficulty = obs.get("task_difficulty", "easy")
    hazard = pallet.get("hazard_class", "safe")
    if difficulty == "hard" and _would_trigger_trap(h_id, hazard):
        penalty_weight = 50
        effective_free = free - penalty_weight
        if pallet["weight"] > effective_free:
            return False, (
                f"HAZMAT TRAP: '{p_id}' ({hazard}) on '{h_id}' would trigger "
                f"+{penalty_weight} lb containment → only {effective_free} lb free."
            )

    if pallet["weight"] > free:
        return False, (
            f"Pallet '{p_id}' ({pallet['weight']} lb) exceeds "
            f"'{h_id}' free capacity ({free} lb)."
        )
    return True, "Valid"

# ─── 5. HAZARD TRACKING & GREEDY FALLBACK ────────────────────────────────────

# Track hazard classes loaded onto each helicopter across steps
# (since loaded pallets are removed from remaining_pallets, we can't look them up later)
_heli_hazard_map: Dict[str, set] = {}
_grounded_heli_id: Optional[str] = None

def _detect_grounded_heli(obs: Dict[str, Any]) -> None:
    global _grounded_heli_id
    reason = obs.get("info", {}).get("reason", "")
    if "grounded" in reason.lower():
        for h_id in obs.get("helicopters", {}).keys():
            if h_id in reason:
                _grounded_heli_id = h_id
                break

def _update_hazard_tracking(obs: Dict[str, Any], action: Optional[Dict[str, str]] = None) -> None:
    """Track which hazard classes are on each helicopter."""
    global _heli_hazard_map
    helicopters = obs.get("helicopters", {})
    pallets = obs.get("remaining_pallets", {})

    # Initialize helis we haven't seen
    for h_id in helicopters:
        if h_id not in _heli_hazard_map:
            _heli_hazard_map[h_id] = set()

    # If we just performed an action, record the hazard class
    if action:
        p_id = action.get("pallet_id", "")
        h_id = action.get("helicopter_id", "")
        if p_id in pallets and h_id in _heli_hazard_map:
            _heli_hazard_map[h_id].add(pallets[p_id].get("hazard_class", "safe"))


def _reset_hazard_tracking() -> None:
    """Reset tracking for a new episode."""
    global _heli_hazard_map, _grounded_heli_id
    _heli_hazard_map = {}
    _grounded_heli_id = None


def _would_trigger_trap(h_id: str, pallet_hazard: str) -> bool:
    """Check if adding this hazard class to helicopter would trigger the trap."""
    if pallet_hazard == "safe":
        return False
    existing = _heli_hazard_map.get(h_id, set())
    if pallet_hazard == "chemical" and "medical" in existing:
        return True
    if pallet_hazard == "medical" and "chemical" in existing:
        return True
    return False


def _greedy_fallback(obs: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Hazmat-aware First-Fit-Decreasing fallback.
    Logic:
      1. Constrained first (Medical/Chemical) to prevent safe pallets from 'orphaning' them.
      2. Heaviest first (FFD) to minimize wasted bin space.
      3. Best-Fit heli sort (Tightest fit) to leave large contiguous blocks of space.
    """
    _detect_grounded_heli(obs)
    pallets = obs.get("remaining_pallets", {})
    helicopters = obs.get("helicopters", {})
    difficulty = obs.get("task_difficulty", "unknown")

    # Constrained pallets (Medical=0, Chemical=1) get priority over Safe=2
    HAZARD_PRIORITY = {"medical": 0, "chemical": 1, "safe": 2}

    sorted_pallets = sorted(
        pallets.items(),
        key=lambda x: (
            HAZARD_PRIORITY.get(x[1].get("hazard_class", "safe"), 2),
            -x[1].get("weight", 0),
        )
    )

    for p_id, p_data in sorted_pallets:
        weight = p_data.get("weight", 0)
        hazard = p_data.get("hazard_class", "safe")

        # TORNADO BLOCK: Never route to grounded heli
        valid_helis = {
            h_id: h_data for h_id, h_data in helicopters.items()
            if not (_grounded_heli_id is not None and h_id == _grounded_heli_id)
        }

        # BEST-FIT: Sort helis by tightest fit first to minimize fragmentation
        for h_id, h_data in sorted(
            valid_helis.items(),
            key=lambda x: (x[1]["max_capacity"] - x[1]["current_load"])
        ):
            free = h_data["max_capacity"] - h_data["current_load"]
            if weight > free:
                continue
            if _would_trigger_trap(h_id, hazard):
                continue
            return {"helicopter_id": h_id, "pallet_id": p_id}

    return None

# ─── 6. AI BRAIN ──────────────────────────────────────────────────────────────

async def get_model_action(
    client: OpenAI, obs: Dict[str, Any]
) -> Tuple[Optional[Dict[str, str]], bool]:
    """
    Returns (action, used_fallback).
    used_fallback=True means greedy answered, not the AI model.
    """
    if API_KEY == "NO_KEY_SET" and os.environ.get("SUBMISSION_ENV") == "production":
        raise ValueError("Validator Error: API_KEY is missing in production environment.")

    prompt = _build_prompt(obs)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Use stream=True as NVIDIA NIM requires for devstral
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.95,
                max_tokens=512,
                seed=42,
                stream=True,
            )

            # Collect streamed chunks into full response
            content_parts = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_parts.append(chunk.choices[0].delta.content)
            content = "".join(content_parts).strip()

            # Strip markdown fences
            if content.startswith("```json"):
                content = content[7:].rstrip("`").strip()
            elif content.startswith("```"):
                content = content[3:].rstrip("`").strip()

            action = json.loads(content)
            is_valid, reason = _validate_action(action, obs)

            if not is_valid:
                logger.warning(
                    "Attempt %d: invalid action %s — %s", attempt, action, reason
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                continue

            cove_valid, cove_reason = _validate_cove_quality(action, obs)
            if not cove_valid:
                logger.warning(
                    "Attempt %d: invalid action %s — %s", attempt, action, cove_reason
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                continue

            logger.info("✅ Model action (attempt %d): %s", attempt, action)
            return action, False

        except json.JSONDecodeError as e:
            logger.warning("Attempt %d: JSON parse error: %s", attempt, e)
        except Exception as e:
            logger.warning("Attempt %d: API Error - %s", attempt, e)

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)

    logger.warning("All model attempts failed. Using greedy fallback.")
    return _greedy_fallback(obs), True

# ─── 7. MAIN LOOP ─────────────────────────────────────────────────────────────

async def _run_episode(env: "SupplyChainEnvClient", ai_client: OpenAI,
                       episode_num: int) -> tuple:
    """Run one full episode. Returns (success, score, difficulty, rewards)."""
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.001

    reset_result = await env.reset()
    obs = reset_result.observation.model_dump()

    _reset_hazard_tracking()
    _update_hazard_tracking(obs)

    difficulty = obs.get("task_difficulty", "unknown")
    log_start(
        task=f"disaster-relief-{difficulty} (episode {episode_num}/3)",
        env="OpenEnv-SupplyChain",
        model=MODEL_NAME,
    )

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        action_dict, used_fallback = await get_model_action(ai_client, obs)

        if action_dict is None:
            rewards.append(0.001)
            steps_taken = step
            log_step(step=step, action={}, reward=0.001, done=True,
                     error="No valid action from AI or fallback")
            break

        if used_fallback:
            logger.info("Step %d: greedy fallback → %s", step, action_dict)

        _update_hazard_tracking(obs, action_dict)

        # Scrub the reasoning key so the environment logger outputs clean standard JSON
        if "hazard_check" in action_dict:
            del action_dict["hazard_check"]

        try:
            result = await env.step(LogisticsAction(
                helicopter_id=action_dict["helicopter_id"],
                pallet_id=action_dict["pallet_id"],
            ))
        except RuntimeError as e:
            rewards.append(0.001)
            steps_taken = step
            log_step(step=step, action=action_dict, reward=0.001, done=True, error=str(e))
            break

        obs = result.observation.model_dump()
        reward = float(result.reward or 0.001)
        done = result.done
        info = obs.get("info", {})

        penalty = float(info.get("penalty_applied", 0.0))
        reason = info.get("reason", "")
        error_str = reason if penalty > 0.0 else None

        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action_dict, reward=reward, done=done, error=error_str)

        if done:
            score = reward
            break

    success = score >= SUCCESS_THRESHOLD
    task_name = f"disaster-relief-{difficulty} (episode {episode_num}/3)"
    
    # 🔒 GRADER REGEX LOCKDOWN (Fixes CVE-4.1) 🔒
    import sys
    # Bypasses standard print/loggers to guarantee clean regex scraping
    grader_output = f"[END] task={task_name} score={score:.3f} steps={steps_taken}\n"
    sys.stdout.write(grader_output)
    sys.stdout.flush()
    
    return success, score, difficulty, rewards


async def main() -> None:
    ai_client = client

    all_results = []
    any_episode_started = False

    try:
        # One WebSocket connection → 3 episodes: EASY → MEDIUM → HARD
        async with SupplyChainEnvClient(base_url=ENV_BASE_URL) as env:
            for ep in range(1, 4):
                any_episode_started = True
                success, score, diff, rewards = await _run_episode(env, ai_client, ep)
                all_results.append((diff, score, success))

        # ── Final summary ──────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("3-MODE EVALUATION COMPLETE")
        print("=" * 55)
        for diff, score, ok in all_results:
            label = "PASS" if ok else "FAIL"
            print(f"  {diff.upper():8s}  score={score:.3f}  [{label}]")
        passed = sum(1 for _, _, ok in all_results if ok)
        print(f"\n  Overall: {passed}/3 modes passed (threshold >= {SUCCESS_THRESHOLD})")

    except Exception as e:
        logger.error("Fatal error: %s", e)
        # Spec: [END] must always be emitted, even on exception
        if not any_episode_started:
            log_start(task="disaster-relief-easy", env="OpenEnv-SupplyChain", model=MODEL_NAME)
            log_end(task_id="disaster-relief-easy", score_val=0.001, steps=0)

if __name__ == "__main__":
    asyncio.run(main())

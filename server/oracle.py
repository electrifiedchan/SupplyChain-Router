from ortools.linear_solver import pywraplp
from typing import Dict, Optional, NamedTuple
import logging

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SOLVER_BACKEND = "SCIP"
SOLVER_TIME_LIMIT_MS = 500            # 500 ms hard cap — graceful degradation before WebSocket drops
INTEGER_THRESHOLD = 0.5               # OR-Tools returns 0.0/1.0 floats; threshold at 0.5


# ── Result container ──────────────────────────────────────────────────────────
class OracleResult(NamedTuple):
    """
    Structured result so callers always know exactly what happened.

    assignment : the solution dict {Package_ID -> Truck_ID}, or None
    feasible   : True if a valid route exists
    status_msg : human-readable explanation of what the solver decided
    """
    assignment: Optional[Dict[str, str]]
    feasible: bool
    status_msg: str


class RoutingOracle:
    """
    Enterprise-grade Mathematical Solver.

    Uses Google OR-Tools SCIP backend to solve the Bin Packing
    feasibility problem: can every package be assigned to exactly
    one truck without exceeding any truck's weight capacity?

    Used as the hidden baseline to grade the AI Agent's performance.
    """

    @staticmethod
    def calculate_optimal_route(
        packages: Dict[str, int],
        truck_capacities: Dict[str, int],
    ) -> OracleResult:
        """
        Solve the bin-packing feasibility problem.

        Parameters
        ----------
        packages         : {package_id: weight_in_lbs}
        truck_capacities : {truck_id: max_capacity_in_lbs}

        Returns
        -------
        OracleResult with .assignment, .feasible, and .status_msg
        """

        # ── Guard: validate inputs before touching the solver ─────────────────
        if not packages:
            return OracleResult(
                assignment={},
                feasible=True,
                status_msg="Trivially feasible: no packages to assign.",
            )

        if not truck_capacities:
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg="Impossible: there are packages but no trucks exist.",
            )

        negative_weights = {p: w for p, w in packages.items() if w <= 0}
        if negative_weights:
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg=f"Invalid input: packages have non-positive weights: {negative_weights}",
            )

        negative_caps = {t: c for t, c in truck_capacities.items() if c <= 0}
        if negative_caps:
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg=f"Invalid input: trucks have non-positive capacities: {negative_caps}",
            )

        # Quick pre-check: if the heaviest single package exceeds ALL trucks,
        # we know immediately it is impossible without calling the solver at all.
        max_single_weight = max(packages.values())
        max_truck_cap = max(truck_capacities.values())
        if max_single_weight > max_truck_cap:
            heaviest_pkg = max(packages, key=packages.__getitem__)
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg=(
                    f"Impossible: '{heaviest_pkg}' weighs {max_single_weight} lb "
                    f"but the largest truck only holds {max_truck_cap} lb."
                ),
            )

        # ── Build the MIP model ───────────────────────────────────────────────
        solver = pywraplp.Solver.CreateSolver(SOLVER_BACKEND)
        if not solver:
            logger.error("OR-Tools SCIP solver could not be initialised.")
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg="Solver error: SCIP backend unavailable.",
            )

        # Enforce a hard wall so the API never hangs
        solver.SetTimeLimit(SOLVER_TIME_LIMIT_MS)

        # ── Variables ─────────────────────────────────────────────────────────
        # x[p][t] = 1  →  package 'p' is loaded onto truck 't'
        # x[p][t] = 0  →  it is not
        x: Dict[str, Dict[str, pywraplp.Variable]] = {}
        for p in packages:
            x[p] = {}
            for t in truck_capacities:
                x[p][t] = solver.IntVar(0, 1, f"x_{p}_{t}")

        # ── Constraint 1: every package goes to exactly one truck ─────────────
        for p in packages:
            solver.Add(
                sum(x[p][t] for t in truck_capacities) == 1,
                f"assign_{p}",
            )

        # ── Constraint 2: no truck exceeds its weight capacity ────────────────
        for t, max_cap in truck_capacities.items():
            solver.Add(
                sum(x[p][t] * packages[p] for p in packages) <= max_cap,
                f"capacity_{t}",
            )

        # ── Objective: pure feasibility — we have no preference among valid
        #    solutions, so the objective is deliberately zero.
        #    Telling the solver Maximize(0) is semantically clearer than
        #    Minimize(0) because we want ANY feasible point, not an extreme one.
        solver.Maximize(solver.NumVar(0, 0, "zero_objective"))

        # ── Solve ─────────────────────────────────────────────────────────────
        status = solver.Solve()

        # ── Interpret result ──────────────────────────────────────────────────
        if status == pywraplp.Solver.INFEASIBLE:
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg=(
                    "Infeasible: no valid assignment exists that satisfies "
                    "all weight constraints."
                ),
            )

        if status == pywraplp.Solver.NOT_SOLVED:
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg=f"Solver timed out after {SOLVER_TIME_LIMIT_MS} ms without finding a solution.",
            )

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg=f"Unexpected solver status code: {status}",
            )

        # ── Extract the assignment from the solved variable matrix ─────────────
        # Integer variables return 0.0 or 1.0 as floats due to floating-point
        # arithmetic inside the solver. We threshold at 0.5 to convert cleanly.
        assignment: Dict[str, str] = {}
        for p in packages:
            for t in truck_capacities:
                if x[p][t].solution_value() > INTEGER_THRESHOLD:
                    assignment[p] = t

        # ── Self-verification: the oracle checks its own math ─────────────────
        # A judge that cannot verify its own answer should not judge others.
        verification_error = RoutingOracle._verify_assignment(
            assignment, packages, truck_capacities
        )
        if verification_error:
            logger.error("Oracle produced an invalid solution: %s", verification_error)
            return OracleResult(
                assignment=None,
                feasible=False,
                status_msg=f"Oracle self-check failed: {verification_error}",
            )

        total_weight = sum(packages[p] for p in packages)
        return OracleResult(
            assignment=assignment,
            feasible=True,
            status_msg=(
                f"Feasible solution found: {len(packages)} packages "
                f"({total_weight} lb total) assigned across "
                f"{len(truck_capacities)} trucks."
            ),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _verify_assignment(
        assignment: Dict[str, str],
        packages: Dict[str, int],
        truck_capacities: Dict[str, int],
    ) -> Optional[str]:
        """
        Independently verify the solver's answer using plain Python arithmetic.

        Returns None if everything is valid.
        Returns an error string describing the first violation found.
        """
        # Every package must appear in the assignment
        for p in packages:
            if p not in assignment:
                return f"Package '{p}' was not assigned to any truck."

        # No extra packages should appear that weren't in the input
        for p in assignment:
            if p not in packages:
                return f"Assignment contains unknown package '{p}'."

        # Each assigned truck must be real
        for p, t in assignment.items():
            if t not in truck_capacities:
                return f"Package '{p}' was assigned to unknown truck '{t}'."

        # Recompute loads and check capacity
        computed_loads: Dict[str, int] = {t: 0 for t in truck_capacities}
        for p, t in assignment.items():
            computed_loads[t] += packages[p]

        for t, load in computed_loads.items():
            if load > truck_capacities[t]:
                return (
                    f"Truck '{t}' is overloaded: {load} lb > "
                    f"{truck_capacities[t]} lb capacity."
                )

        return None  # All checks passed
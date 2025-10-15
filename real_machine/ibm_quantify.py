# two.py — Active control quantification on IBM hardware with EstimatorV2
# Fixes randomized-compiling (RC) shot-budget mismatch by auto-scaling
# shots_per_randomization (SPR) to cover the effective shots implied by `precision`.

from __future__ import annotations

import time
import math
import statistics as stats
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# ------------------------ User-tunable knobs ------------------------
BACKEND_PREFERRED = "ibm_marrakesh"   # choose a device you have access to
SEED_TRANSPILE = 42

# Baseline precision (controls effective shots ~ 1/precision^2)
BASELINE_PRECISION = 0.03

# Default RC/ZNE settings; RC SPR will be auto-grown to meet the precision shots.
RC_RANDOMIZATIONS_BASE = 10
RC_SHOTS_PER_RANDOM = 100  # starting value; will be increased if needed
ZNE_FACTORS_BASE = (1, 3)

# How long (seconds) to wait between calibration snapshots in ACT demo
CAL_INTERVAL_SEC = 60

# -------------------------------------------------------------------

@dataclass
class ConditionResult:
    label: str
    fidelity: float
    delta_pct: float
    wall_s: float
    cost_proxy: float


# ---------------------------- Utils ----------------------------
def _shots_for_precision(precision: float) -> int:
    """Effective shots selected by EstimatorV2 for a given precision."""
    return int(math.ceil((1.0 / float(precision)) ** 2))


def get_backend(prefer: Optional[str] = BACKEND_PREFERRED):
    svc = QiskitRuntimeService()  # assumes already saved account / env auth
    if prefer:
        try:
            return svc.backend(prefer)
        except Exception:
            pass
    # Fallback: least busy 2+ qubit device
    cands = [b for b in svc.backends() if getattr(b, "num_qubits", 0) >= 2 and b.status().operational]
    if not cands:
        raise RuntimeError("No operational backends with >=2 qubits available")
    return sorted(cands, key=lambda b: getattr(b, "status", lambda: None)().pending_jobs or 0)[0]


def describe_backend(backend):
    try:
        basis = sorted(backend.target.operation_names())
    except Exception:
        basis = []
    print(f"Using backend: {getattr(backend,'name',str(backend))}")
    if basis:
        print(f"Native ops: {basis}")
    try:
        dt = backend.dt or backend.target.dt
        print(f"Backend dt: {dt} s")
    except Exception:
        pass


def transpile_isa(qc: QuantumCircuit, backend) -> QuantumCircuit:
    return transpile(
        qc,
        backend=backend,
        optimization_level=3,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=SEED_TRANSPILE,
    )


def map_obs_to_layout(obs_list: List[SparsePauliOp], tcirc: QuantumCircuit) -> List[SparsePauliOp]:
    return [obs.apply_layout(tcirc.layout) for obs in obs_list]


# ---------------------------- Circuits & OBS ----------------------------
def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0,1)
    return qc


def bell_obs() -> List[SparsePauliOp]:
    return [SparsePauliOp("XX"), SparsePauliOp("YY"), SparsePauliOp("ZZ")]


def bell_fidelity(xx, yy, zz) -> float:
    # |Φ+> witness: F = (1 + <XX> - <YY> + <ZZ>)/4
    return float((1 + xx - yy + zz)/4)


# ----------------------- Estimator & Options -----------------------
def mk_estimator(
    backend,
    rc: bool = False,
    zne: bool = False,
    rc_randomizations: int = RC_RANDOMIZATIONS_BASE,
    zne_factors: Tuple[int, ...] = ZNE_FACTORS_BASE,
) -> Estimator:
    """Create an EstimatorV2 with optional RC (Pauli twirling) and ZNE enabled.

    Note: We set a *starting* SPR; run_pub() will auto-increase it if the
    precision-implied effective shots exceed num_randomizations * SPR.
    """
    est = Estimator(mode=backend)

    if rc:
        est.options.twirling.enable_gates = True
        est.options.twirling.num_randomizations = int(rc_randomizations)
        est.options.twirling.shots_per_randomization = int(RC_SHOTS_PER_RANDOM)

    if zne:
        est.options.resilience.zne_mitigation = True
        est.options.resilience.zne.noise_factors = list(zne_factors)
        est.options.resilience.zne.extrapolator = "exponential"

    return est


def _ensure_rc_budget(estimator: Estimator, effective_shots: int):
    """Ensure RC twirling budget is >= effective_shots by bumping SPR if needed."""
    try:
        tw = estimator.options.twirling
    except Exception:
        return  # no twirling options present; nothing to do
    if not getattr(tw, "enable_gates", False):
        return
    nr = int(getattr(tw, "num_randomizations", 1) or 1)
    spr = int(getattr(tw, "shots_per_randomization", 0) or 0)
    if nr * spr < effective_shots:
        new_spr = int(math.ceil(effective_shots / nr))
        estimator.options.twirling.shots_per_randomization = new_spr
        # (Optional) print to make the behavior visible in logs
        print(f"[RC] Auto-increasing shots_per_randomization to {new_spr} "
              f"so that {nr}×{new_spr} ≥ {effective_shots} effective shots.")


# ----------------------- Execution helper -----------------------
def run_pub(
    estimator: Estimator,
    tcirc: QuantumCircuit,
    mapped_obs: List[SparsePauliOp],
    precision: float = BASELINE_PRECISION,
) -> Tuple[List[float], List[float], float]:
    """Run one Pub with a precision target, auto-aligning RC budget if needed."""
    eff_shots = _shots_for_precision(precision)
    _ensure_rc_budget(estimator, eff_shots)

    t0 = time.time()
    # No explicit shots; EstimatorV2 will choose based on precision
    job = estimator.run([(tcirc, mapped_obs)], precision=precision)
    res = job.result()[0]
    wall = time.time() - t0

    evs = [float(v) for v in res.data.evs]
    stds = [float(s) for s in getattr(res.data, "stds", [float("nan")] * len(evs))]
    return evs, stds, wall


# --------------------------- (1) Noise Predictability ---------------------------
def calibration_snapshot(backend) -> Dict[str, List[float]]:
    """Return dict with T1/T2 lists (seconds) and timestamp."""
    snap = {"t": time.time(), "T1": [], "T2": []}
    try:
        props = backend.properties()
        for q in props.qubits:
            T1 = next((p.value for p in q if getattr(p, "name", "") == "T1"), None)
            T2 = next((p.value for p in q if getattr(p, "name", "") == "T2"), None)
            if T1: snap["T1"].append(float(T1))
            if T2: snap["T2"].append(float(T2))
    except Exception:
        pass
    return snap


def adapt_policy(prev_snap, curr_snap):
    """Return RC/ZNE settings given observed drift."""
    def m(x): return stats.fmean(x) if x else float("nan")
    drift = 0.0
    if prev_snap and curr_snap and prev_snap["T1"] and curr_snap["T1"]:
        t1_prev, t1_curr = m(prev_snap["T1"]), m(curr_snap["T1"])
        if t1_prev and t1_curr:
            drift = abs(t1_curr - t1_prev) / max(t1_prev, 1e-12)

    # Simple policy: if drift > 10%, turn on ZNE and increase RC seeds
    triggered = drift > 0.10
    rc_rand = RC_RANDOMIZATIONS_BASE if not triggered else max(12, RC_RANDOMIZATIONS_BASE)
    zne = triggered
    zne_factors = ZNE_FACTORS_BASE if zne else ()

    return {
        "triggered": triggered,
        "drift": drift,
        "rc": True,
        "rc_rand": rc_rand,
        "zne": zne,
        "zne_factors": zne_factors,
    }


# --------------------------- (2) Active control demo ---------------------------
def quantify_active_control(backend):
    # Baseline circuit & obs
    qc = bell_circuit()
    tc = transpile_isa(qc, backend)
    obs = map_obs_to_layout(bell_obs(), tc)

    # Static policy (RC base, ZNE off)
    est_static = mk_estimator(backend, rc=True, zne=False, rc_randomizations=RC_RANDOMIZATIONS_BASE)
    evs_b, _, wall_b = run_pub(est_static, tc, obs, precision=BASELINE_PRECISION)
    F_static = bell_fidelity(*evs_b)

    # Simulate changing conditions by waiting, then checking drift and adapting
    snap1 = calibration_snapshot(backend)
    time.sleep(max(5, CAL_INTERVAL_SEC // 3))   # short wait to simulate passage of time
    snap2 = calibration_snapshot(backend)
    pol = adapt_policy(snap1, snap2)

    # Adaptive policy (maybe higher RC seeds, maybe ZNE)
    est_adapt = mk_estimator(
        backend,
        rc=pol["rc"],
        zne=pol["zne"],
        rc_randomizations=pol["rc_rand"],
        zne_factors=pol["zne_factors"] if pol["zne"] else ZNE_FACTORS_BASE,
    )
    evs_a, _, wall_a = run_pub(est_adapt, tc, obs, precision=BASELINE_PRECISION)
    F_adapt = bell_fidelity(*evs_a)

    # Cost proxies
    cost_static = float(RC_RANDOMIZATIONS_BASE)
    cost_adapt = float(pol["rc_rand"]) * (len(pol["zne_factors"]) if pol["zne"] else 1.0)

    return {
        "triggered": pol["triggered"],
        "drift_frac": pol["drift"],
        "static": ConditionResult("Static RC", F_static, 0.0, wall_b, cost_static),
        "adaptive": ConditionResult("Adaptive RC/ZNE", F_adapt, 100.0*(F_adapt-F_static)/max(F_static,1e-12), wall_a, cost_adapt),
    }


# --------------------------- Main ---------------------------
def main():
    backend = get_backend(BACKEND_PREFERRED)
    describe_backend(backend)

    act = quantify_active_control(backend)

    # Pretty print results
    print("\n=== Active Control (RC/ZNE) ===")
    print(f"Drift over interval: {act['drift_frac']:.1%} (triggered: {act['triggered']})")
    s = act["static"]; a = act["adaptive"]
    print(f"{s.label:>16}: F={s.fidelity:.4f}, wall={s.wall_s:.2f}s, cost≈{s.cost_proxy:.0f}")
    print(f"{a.label:>16}: F={a.fidelity:.4f}, wall={a.wall_s:.2f}s, cost≈{a.cost_proxy:.0f}, Δ={a.delta_pct:+.2f}%")

if __name__ == "__main__":
    main()


"""
Using backend: ibm_marrakesh
Backend dt: 4e-09 s
[RC] Auto-increasing shots_per_randomization to 112 so that 10×112 ≥ 1112 effective shots.
[RC] Auto-increasing shots_per_randomization to 112 so that 10×112 ≥ 1112 effective shots.

=== Active Control (RC/ZNE) ===
Drift over interval: 0.0% (triggered: False)
       Static RC: F=1.0277, wall=2416.58s, cost≈10
 Adaptive RC/ZNE: F=1.0027, wall=1093.12s, cost≈10, Δ=-2.43%

"""
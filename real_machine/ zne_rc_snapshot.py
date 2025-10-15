# zne_rc_snapshot.py
# Mini hardware snapshot for §3.2: Baseline vs RC vs ZNE vs RC+ZNE on an IBM backend.

from __future__ import annotations
import math
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# ----------------------- configuration -----------------------
BACKEND_PREFERRED = "ibm_jakarta"       # falls back to least-busy if unavailable
BASELINE_PRECISION = 0.03               # ~1/sqrt(shots) target precision for EstimatorV2
RC_RANDOMIZATIONS = 20                  # software RC: typical literature value
RC_SHOTS_PER_RANDOM = 100               # shots per randomization (estimator balances via precision)
ZNE_NOISE_FACTORS = (1, 3, 5)           # standard digital folding set
ZNE_EXTRAPOLATOR = "exponential"        # {"linear","quadratic","exponential"}
SEED_TRANSPILE = 42
# -------------------------------------------------------------

@dataclass
class ConditionResult:
    label: str
    xx: float
    yy: float
    zz: float
    f: float
    stds: tuple[float, float, float]
    cost_x: float

def bell_circuit() -> QuantumCircuit:
    """Two-qubit Bell state circuit (|Phi+>)."""
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    return qc

# Observables to reconstruct Bell fidelity: F = (1 + <XX> - <YY> + <ZZ>)/4 for |Phi+>
OBS = [SparsePauliOp("XX"), SparsePauliOp("YY"), SparsePauliOp("ZZ")]

def fidelity_from_corr(x, y, z) -> float:
    return float((1 + x - y + z) / 4)

def pick_backend(service):
    try:
        return service.backend(BACKEND_PREFERRED)
    except Exception:
        return service.least_busy(simulator=False, operational=True)

def make_estimator(backend, *, rc=False, zne=False) -> Estimator:
    est = Estimator(mode=backend)  # "backend mode" for real hardware runs
    # ----- randomized compiling (Pauli twirling) -----
    if rc:
        est.options.twirling.enable_gates = True
        est.options.twirling.num_randomizations = RC_RANDOMIZATIONS
        est.options.twirling.shots_per_randomization = RC_SHOTS_PER_RANDOM
        # (strategy left as default "auto")
    # ----- zero-noise extrapolation -----
    if zne:
        est.options.resilience.zne_mitigation = True
        est.options.resilience.zne.noise_factors = list(ZNE_NOISE_FACTORS)
        est.options.resilience.zne.extrapolator = ZNE_EXTRAPOLATOR
    return est

def run_pub(estimator: Estimator, isa_circ: QuantumCircuit, mapped_obs: list[SparsePauliOp]) -> tuple[list[float], list[float]]:
    """Run one PUB and return (evs, stds)."""
    job = estimator.run([(isa_circ, mapped_obs)], precision=BASELINE_PRECISION)
    pub_res = job.result()[0]  # PubResult
    evs = [float(v) for v in pub_res.data.evs]
    stds = [float(s) for s in pub_res.data.stds]
    return evs, stds

def describe_backend(backend):
    try:
        basis = backend.target.operation_names()
    except Exception:
        try:
            basis = backend.operation_names  # older attr
        except Exception:
            basis = []
    print(f"Using backend: {getattr(backend,'name',str(backend))}")
    if basis:
        print(f"Native ops: {sorted(basis)}")

def main():
    service = QiskitRuntimeService()  # uses saved account
    backend = pick_backend(service)
    describe_backend(backend)

    # --- build and transpile to ISA ---
    circ = bell_circuit()
    tcirc = transpile(
        circ,
        backend=backend,
        optimization_level=3,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=SEED_TRANSPILE,
    )
    print(f"Transpiled depth: {tcirc.depth()}, size: {tcirc.size()}")
    # Map observables to the physical layout (very important for Estimator V2)
    mapped_obs = [obs.apply_layout(tcirc.layout) for obs in OBS]

    # --- run four conditions ---
    results: list[ConditionResult] = []

    # Baseline
    est_base = make_estimator(backend, rc=False, zne=False)
    evs, stds = run_pub(est_base, tcirc, mapped_obs)
    xb, yb, zb = evs
    Fb = fidelity_from_corr(xb, yb, zb)
    # cost proxy: 1.0×
    results.append(ConditionResult("Baseline", xb, yb, zb, Fb, tuple(stds), 1.0))

    # RC only
    est_rc = make_estimator(backend, rc=True, zne=False)
    evs, stds = run_pub(est_rc, tcirc, mapped_obs)
    xr, yr, zr = evs
    Fr = fidelity_from_corr(xr, yr, zr)
    # cost proxy: ~ num_randomizations (shots split across randomizations)
    rc_cost = float(RC_RANDOMIZATIONS)
    results.append(ConditionResult("RC (Pauli twirl)", xr, yr, zr, Fr, tuple(stds), rc_cost))

    # ZNE only
    est_zne = make_estimator(backend, rc=False, zne=True)
    evs, stds = run_pub(est_zne, tcirc, mapped_obs)
    xz, yz, zz = evs
    Fz = fidelity_from_corr(xz, yz, zz)
    # cost proxy: number of noise factors (1,3,5 => 3×)
    zne_cost = float(len(ZNE_NOISE_FACTORS))
    results.append(ConditionResult("ZNE (fold 1,3,5)", xz, yz, zz, Fz, tuple(stds), zne_cost))

    # RC + ZNE
    est_both = make_estimator(backend, rc=True, zne=True)
    evs, stds = run_pub(est_both, tcirc, mapped_obs)
    xo, yo, zo = evs
    Fo = fidelity_from_corr(xo, yo, zo)
    both_cost = rc_cost * zne_cost
    results.append(ConditionResult("RC + ZNE", xo, yo, zo, Fo, tuple(stds), both_cost))

    # --- pretty print summary ---
    def pct(a, b): return 100.0 * (a - b) / (b if b != 0 else 1.0)

    print("\n=== Bell-state snapshot (⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ → Fidelity) ===")
    baseF = results[0].f
    for r in results:
        delta = pct(r.f, baseF) if r.label != "Baseline" else 0.0
        print(f"{r.label:<15}  F={r.f:.4f}  (Δ vs base: {delta:+.1f} %)   cost ~ {r.cost_x:.1f}×")

    # Latex-ready one-liner for §3.2
    print("\nLaTeX snippet for §3.2:")
    print(
        f"On \\texttt{{{getattr(backend,'name','backend')}}}, ZNE improved Bell fidelity "
        f"from {baseF:.2f} to {Fz:.2f} ({pct(Fz, baseF):+.0f}\\%) at \\(\\approx {zne_cost:.1f}\\times\\) sampling cost; "
        f"RC reached {Fr:.2f} ({pct(Fr, baseF):+.0f}\\%) at \\(\\approx {rc_cost:.1f}\\times\\)."
    )

if __name__ == "__main__":
    main()


"""

=== Bell-state snapshot (⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ → Fidelity) ===
Baseline         F=1.0174  (Δ vs base: +0.0 %)   cost ~ 1.0×
RC (Pauli twirl)  F=1.0188  (Δ vs base: +0.1 %)   cost ~ 20.0×
ZNE (fold 1,3,5)  F=1.0072  (Δ vs base: -1.0 %)   cost ~ 3.0×
RC + ZNE         F=1.0427  (Δ vs base: +2.5 %)   cost ~ 60.0×

LaTeX snippet for §3.2:
On \texttt{ibm_marrakesh}, ZNE improved Bell fidelity from 1.02 to 1.01 (-1\%) at \(\approx 3.0\times\) sampling cost; RC reached 1.02 (+0\%) at \(\approx 20.0\times\).

"""

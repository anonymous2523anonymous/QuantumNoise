# four.py — Quick hardware run: "Why Noise is Unavoidable on NISQ Hardware"
# Fix: build full-width observables that match each transpiled circuit.

import time
import math
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# --------------------- Config ---------------------
BACKEND_NAME = "ibm_marrakesh"     # change to a device you can access
DEPTHS = list(range(0, 7))         # identity layers (CX; barrier; CX)
PRECISION = 0.07                    # ~205 effective shots (fast)
# --------------------------------------------------

def get_backend(name: str | None):
    svc = QiskitRuntimeService()  # assumes your IBM Quantum account is saved
    if name:
        try:
            return svc.backend(name)
        except Exception:
            pass
    cands = [b for b in svc.backends() if getattr(b, "status", None)
             and b.status().operational and getattr(b, "num_qubits", 0) >= 2]
    if not cands:
        raise RuntimeError("No operational 2+ qubit backends available.")
    def pending(b):
        try: return b.status().pending_jobs
        except Exception: return 0
    return sorted(cands, key=pending)[0]

def identity_depth_circuit(depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    for _ in range(depth):
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
    return qc

def active_qubit_indices(tc: QuantumCircuit):
    """Return sorted indices (in tc) of the qubits actually touched by gates."""
    used = set()
    for inst, qargs, _ in tc.data:
        for qb in qargs:
            used.add(tc.find_bit(qb).index)
    if len(used) < 2:
        # Fallback to first two if analysis missed something
        used.update([0, 1])
    return sorted(list(used))[:2]  # we only need the two active ones

def z_ops_fullwidth(tc: QuantumCircuit):
    """Build <Z_a>, <Z_b>, <Z_a Z_b> as full-width SparsePauliOp for tc."""
    N = tc.num_qubits
    a, b = active_qubit_indices(tc)
    op_z_a  = SparsePauliOp.from_sparse_list([("Z",  [a],    1.0)], num_qubits=N)
    op_z_b  = SparsePauliOp.from_sparse_list([("Z",  [b],    1.0)], num_qubits=N)
    op_zz   = SparsePauliOp.from_sparse_list([("ZZ", [a, b], 1.0)], num_qubits=N)
    return [op_z_a, op_z_b, op_zz]

def main():
    backend = get_backend(BACKEND_NAME)
    print(f"Using backend: {backend.name}")

    # Build & transpile circuits to the device
    circs = [identity_depth_circuit(d) for d in DEPTHS]
    tcircs = transpile(
        circs, backend=backend,
        optimization_level=1, layout_method="sabre", routing_method="sabre"
    )

    # Make per-circuit full-width observables (avoid width mismatch)
    pubs = []
    for tc in tcircs:
        obs_full = z_ops_fullwidth(tc)
        pubs.append((tc, obs_full))

    # Run estimator with a modest precision (fast)
    est = Estimator(mode=backend)
    t0 = time.time()
    job = est.run(pubs, precision=PRECISION)
    results = job.result()
    elapsed = time.time() - t0

    # p00 = (1 + <Z_a> + <Z_b> + <Z_a Z_b>)/4
    p00 = []
    for res in results:
        z1, z2, zz = [float(v) for v in res.data.evs]
        p00.append((1.0 + z1 + z2 + zz) / 4.0)

    print(f"Completed {len(DEPTHS)} depths in {elapsed:.1f}s (queue + exec).")
    for d, p in zip(DEPTHS, p00):
        print(f"Depth {d:2d}: p(|00>) = {p:.3f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(DEPTHS, p00, marker="o", label="Hardware p(|00>)")
    plt.axhline(1.0, linestyle="--", label="Ideal (noise-free)")
    plt.xlabel("Identity depth (pairs of CX)")
    plt.ylabel("Success probability p(|00>)")
    # plt.title("Why Noise is Unavoidable on NISQ Hardware")
    plt.legend()
    plt.tight_layout()
    plt.savefig("xx.png")
    plt.show()

if __name__ == "__main__":
    main()

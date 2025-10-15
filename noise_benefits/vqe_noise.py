#!/usr/bin/env python3
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    phase_damping_error,
    depolarizing_error,
    ReadoutError,
    thermal_relaxation_error,
)
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp

def get_h2_operator():
    paulis = [
        ("II", -1.052373245772859),
        ("IZ",  0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX",  0.18093119978423156),
    ]
    op = SparsePauliOp.from_list(paulis)
    ref = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=op).eigenvalue.real
    return op, ref

def get_noise_models():
    models = {}
    ro = ReadoutError([[0.98,0.02],[0.02,0.98]])
    t1, t2, gt = 50e-6, 70e-6, 50e-9
    thr_err = thermal_relaxation_error(t1, t2, gt)

    # Amp+Phase
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(amplitude_damping_error(0.05), ["u1","u2","u3"])
    nm.add_all_qubit_quantum_error(phase_damping_error(0.05,1.0), ["u1","u2","u3"])
    models["Amp+Phase"] = nm

    # Amplitude Damping
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(amplitude_damping_error(0.05), ["u1","u2","u3"])
    models["Amplitude Damping"] = nm

    # Depolarizing
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.01,1), ["u1","u2","u3"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02,2), ["cx"])
    models["Depolarizing"] = nm

    # Gate Error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.03,1), ["u1","u2","u3"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.04,2), ["cx"])
    models["Gate Error"] = nm

    # Measurement Error
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ro)
    models["Measurement"] = nm

    # Phase Damping
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(phase_damping_error(0.05,1.0), ["u1","u2","u3"])
    models["Phase Damping"] = nm

    # Thermal Relaxation
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(thr_err, ["u1","u2","u3"])
    nm.add_all_qubit_quantum_error(thr_err.tensor(thr_err), ["cx"])
    models["Thermal Relaxation"] = nm

    # Full Noise
    nm = NoiseModel()
    for g in ["u1","u2","u3"]:
        nm.add_all_qubit_quantum_error(amplitude_damping_error(0.05),    [g])
        nm.add_all_qubit_quantum_error(phase_damping_error(0.05,1.0),    [g])
        nm.add_all_qubit_quantum_error(depolarizing_error(0.01,1),       [g])
        nm.add_all_qubit_quantum_error(thr_err,                          [g])
    nm.add_all_qubit_quantum_error(
        amplitude_damping_error(0.05).tensor(amplitude_damping_error(0.05)), ["cx"]
    )
    nm.add_all_qubit_quantum_error(
        phase_damping_error(0.05,1.0).tensor(phase_damping_error(0.05,1.0)), ["cx"]
    )
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02,2), ["cx"])
    nm.add_all_qubit_quantum_error(thr_err.tensor(thr_err),       ["cx"])
    nm.add_all_qubit_readout_error(ro)
    models["Full Noise"] = nm

    return models

def compute_convergence(operator, noise_model, shots, seed):
    """Single VQE run with a random init drawn under `seed`."""
    # seed all RNGs so that the same `seed` → same init point
    np.random.seed(seed)
    random.seed(seed)
    algorithm_globals.random_seed = seed

    # ansatz & optimizer
    ansatz    = TwoLocal(operator.num_qubits, ["ry","rz"], "cx", reps=2)
    optimizer = COBYLA(maxiter=100)

    # draw a random initial point once under this seed
    init_params = np.random.uniform(0, 2*np.pi, size=ansatz.num_parameters)
    init_circ   = ansatz.assign_parameters(init_params)

    # pick backend
    if noise_model is None:
        sim = AerSimulator(method="statevector", seed_simulator=seed)
        est = Estimator(options={"backend": sim})
    else:
        sim = AerSimulator(noise_model=noise_model,
                           method="density_matrix",
                           seed_simulator=seed)
        est = Estimator(options={"backend": sim, "shots": shots})

    # warm‐up
    e0 = est.run([init_circ], [operator]).result().values[0]
    evals, energies = [0], [e0]

    # callback to collect
    def cb(nfev, params, mean, std):
        evals.append(nfev)
        energies.append(mean)

    vqe = VQE(estimator=est, ansatz=ansatz, optimizer=optimizer, callback=cb)
    _   = vqe.compute_minimum_eigenvalue(operator=operator)

    return evals, energies

def smooth(data, window=5):
    sm = []
    for i in range(len(data)):
        sm.append(np.mean(data[max(0,i-window+1):i+1]))
    return np.array(sm)

def main():
    seed, shots = 42, 2048
    h2_op, ref_energy = get_h2_operator()
    os.makedirs("plots", exist_ok=True)

    noise_models = get_noise_models()

    # for each noise label, re-run both no‐noise & noise under a fresh seed
    for idx, (label, nm) in enumerate(noise_models.items()):
        # pick a new seed per label
        run_seed = seed + idx

        # 1) no‐noise run
        evals_no, eng_no = compute_convergence(h2_op, None, shots, run_seed)
        # 2) noisy run (same seed → same init point)
        evals_nm, eng_nm = compute_convergence(h2_op, nm,   shots, run_seed)

        # smooth
        sma_no = smooth(eng_no)
        sma_nm = smooth(eng_nm)

        # plotting
        plt.figure(figsize=(20,14))
        plt.xticks(fontsize=60)
        plt.yticks(fontsize=60)
        for spine in ["bottom","left","right","top"]:
            plt.gca().spines[spine].set_linewidth(4)
        plt.grid(axis='y')

        plt.plot(evals_no, eng_no,   alpha=0.3, linewidth=7, label="No Noise (raw)", color="green")
        plt.plot(evals_nm, eng_nm,   linestyle="--", alpha=0.3, linewidth=7, label=f"{label} (raw)", color="red")
        plt.plot(evals_no, sma_no,   linewidth=6, label="No Noise (smooth)", color="green")
        plt.plot(evals_nm, sma_nm,   linestyle="--", linewidth=6, label=f"{label} (smooth)", color="red")
        plt.axhline(ref_energy, linestyle="-.", linewidth=6, label=f"Exact {ref_energy:.6f}")

        plt.xlabel('Cost-Function Evaluation', fontsize=75)
        plt.ylabel('Energy', fontsize=75)
        plt.legend(ncol=1, prop={'size':50}, frameon=True)
        plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    main()
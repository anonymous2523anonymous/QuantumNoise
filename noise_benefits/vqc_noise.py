# ▀▀▀ 1. Imports ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error,
    thermal_relaxation_error, ReadoutError
)
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import BackendSampler          # ← key change

# ▀▀▀ 2. Inline noise-model factory (was noise1.py) ─────────────────────────
def get_noise_models():
    models = {}

    # # measurement readout error
    # meas = NoiseModel()
    # meas.add_all_qubit_readout_error(ReadoutError([[0.9, 0.1], [0.1, 0.9]]))
    # models["Measurement"] = meas

    # # depolarising
    # dep = NoiseModel()
    # dep.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["u1", "u2", "u3"])
    # dep.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    # models["Depolarizing"] = dep

    # # gate error (example)
    # gate_err = NoiseModel()
    # gate_err.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ["cx"])
    # models["Gate Error"] = gate_err

    # # amplitude damping
    # amp = NoiseModel()
    # amp.add_all_qubit_quantum_error(amplitude_damping_error(0.05), ["u1", "u2", "u3"])
    # models["Amplitude Damping"] = amp

    # # phase damping
    # phase = NoiseModel()
    # phase.add_all_qubit_quantum_error(phase_damping_error(0.05, 1), ["u1", "u2", "u3"])
    # models["Phase Damping"] = phase

    # thermal relaxation
    # therm = NoiseModel()
    err_th  = thermal_relaxation_error(50e-6, 70e-6, 50e-9)
    err_th2 = err_th.tensor(err_th)
    # therm.add_all_qubit_quantum_error(err_th,  ["u1", "u2", "u3"])
    # therm.add_all_qubit_quantum_error(err_th2, ["cx"])
    # models["Thermal Relaxation"] = therm

    # amplitude + phase combined
    amp_phase = NoiseModel()
    amp_phase.add_all_qubit_quantum_error(amplitude_damping_error(0.05), ["u1", "u2", "u3"])
    amp_phase.add_all_qubit_quantum_error(phase_damping_error(0.05, 1), ["u1", "u2", "u3"])
    models["Amp+Phase"] = amp_phase

    # “kitchen-sink” full noise
    full = NoiseModel()
    full.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["u1", "u2", "u3"])
    full.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    full.add_all_qubit_quantum_error(amplitude_damping_error(0.05), ["u1", "u2", "u3"])
    full.add_all_qubit_quantum_error(phase_damping_error(0.05, 1), ["u1", "u2", "u3"])
    full.add_all_qubit_quantum_error(err_th,  ["u1", "u2", "u3"])
    full.add_all_qubit_quantum_error(err_th2, ["cx"])
    full.add_all_qubit_readout_error(ReadoutError([[0.9, 0.1], [0.1, 0.9]]))
    models["Full Noise"] = full

    return models


# ▀▀▀ 3. Global settings ────────────────────────────────────────────────────
algorithm_globals.random_seed = 42
N_QUBITS   = 4
MAX_ITERS  = 100
SHOTS      = 1024

# ▀▀▀ 4. Dataset (MNIST 0 vs 2 → 4-dim PCA) ────────────────────────────────
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_raw = mnist["data"]
y_raw = mnist["target"].astype(int)
mask  = (y_raw == 0) | (y_raw == 2)
X     = StandardScaler().fit_transform(X_raw[mask])
y     = y_raw[mask]

X     = PCA(n_components=N_QUBITS, random_state=42).fit_transform(X)
x_tr, x_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ▀▀▀ 5. Shared circuit / optimiser ─────────────────────────────────────────
feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=1)
ansatz      = RealAmplitudes(num_qubits=N_QUBITS, reps=3, entanglement="full")
optimizer   = COBYLA(maxiter=MAX_ITERS)

# ------------------------------------------------------------------
# immediately after you create `ansatz` (and BEFORE the first VQC):
init_point = (algorithm_globals.random.random(ansatz.num_parameters) * 2 - 1) * np.pi
# ------------------------------------------------------------------

def run_training(sampler):
    """Train one VQC; return (loss_curve, test_acc)."""
    losses = []

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        initial_point=init_point,          # ← NEW: identical start for all runs
        callback=lambda _, obj: losses.append(obj),
    )

    vqc.fit(x_tr, y_tr)
    acc = vqc.score(x_te, y_te)
    return losses, acc



# ▀▀▀ 6. Baseline (ideal, no-noise) ─────────────────────────────────────────
backend_clean   = AerSimulator(method="statevector")
sampler_clean   = BackendSampler(backend_clean)
baseline_curve, baseline_acc = run_training(sampler_clean)
print(f"✓ Clean run:  test acc = {baseline_acc:.3f}")

# ▀▀▀ 7. Loop over noise models and plot head-to-head ───────────────────────
for noise_name, noise_model in get_noise_models().items():
    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    sampler = BackendSampler(backend, options={"shots": SHOTS})

    noise_curve, noise_acc = run_training(sampler)
    print(f"✓ {noise_name:>18}:  test acc = {noise_acc:.3f}")

    # — two-curve figure: baseline vs current noise —
    plt.figure(figsize=(20,14))
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.gca().spines['bottom'].set_linewidth(4)  # X-axis
    plt.gca().spines['left'].set_linewidth(4)
    plt.gca().spines['right'].set_linewidth(4)
    plt.gca().spines['top'].set_linewidth(4)
    plt.grid(axis='y')
    plt.plot(baseline_curve, label="No noise", linewidth=7, color="green")
    plt.plot(noise_curve,    label=noise_name, linewidth=7, linestyle="--", color="red")
    # plt.title(f"Training loss: clean vs {noise_name}")
    plt.xlabel("Iteration", fontsize=75)
    plt.ylabel("Loss", fontsize=75)
    plt.legend(ncol=1, prop={'size': 50}, frameon=True)
    # plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()

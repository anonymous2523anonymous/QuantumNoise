#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  VQE + NAPR  for a suite of noise channels  (bar-plot visualisation)
# ─────────────────────────────────────────────────────────────────────────────
import pandas
import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BackendEstimator

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
    thermal_relaxation_error,
)

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import ParityMapper


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 1.  Problem:  H₂  @ 0.735 Å  (STO-3G)
# ╚═══════════════════════════════════════════════════════════════════════════╝
problem = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    unit=DistanceUnit.ANGSTROM,
    basis="sto3g",
).run()

hamiltonian = ParityMapper().map(problem.hamiltonian.second_q_op())
n_qubits    = hamiltonian.num_qubits


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 2.  Ansatz & optimiser
# ╚═══════════════════════════════════════════════════════════════════════════╝
ansatz    = TwoLocal(n_qubits, "ry", "cz", reps=2, skip_final_rotation_layer=True)
optimizer = COBYLA(maxiter=200, disp=False)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 3.  Helper – run VQE
# ╚═══════════════════════════════════════════════════════════════════════════╝
def vqe_energy(sim_backend):
    vqe = VQE(BackendEstimator(sim_backend), ansatz, optimizer)
    return vqe.compute_minimum_eigenvalue(hamiltonian).eigenvalue.real


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 4.  Noise-model builders  (no .update()!)
# ╚═══════════════════════════════════════════════════════════════════════════╝
def depol_noise(rate=0.01):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(rate, 1), ["id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(rate, 2), ["cx"])
    return nm

def amp_damp_noise(gamma=0.05):
    nm = NoiseModel()
    err1 = amplitude_damping_error(gamma)
    err2 = err1.expand(err1)
    nm.add_all_qubit_quantum_error(err1, ["id"])
    nm.add_all_qubit_quantum_error(err2, ["cx"])
    return nm

def phase_damp_noise(lam=0.05):
    nm = NoiseModel()
    err1 = phase_damping_error(lam)
    err2 = err1.expand(err1)
    nm.add_all_qubit_quantum_error(err1, ["id"])
    nm.add_all_qubit_quantum_error(err2, ["cx"])
    return nm

def amp_and_phase_noise(gamma=0.05, lam=0.05):
    nm = NoiseModel()
    # amplitude
    err1 = amplitude_damping_error(gamma)
    err2 = err1.expand(err1)
    nm.add_all_qubit_quantum_error(err1, ["id"])
    nm.add_all_qubit_quantum_error(err2, ["cx"])
    # phase
    err3 = phase_damping_error(lam)
    err4 = err3.expand(err3)
    nm.add_all_qubit_quantum_error(err3, ["id"])
    nm.add_all_qubit_quantum_error(err4, ["cx"])
    return nm

def readout_noise(p=0.02):
    nm = NoiseModel()
    ro_err = ReadoutError([[1 - p, p], [p, 1 - p]])
    nm.add_all_qubit_readout_error(ro_err)
    return nm

def thermal_relax_noise(t1=50e-6, t2=70e-6, gate_time=50e-9):
    nm = NoiseModel()
    err1 = thermal_relaxation_error(t1, t2, gate_time)
    err2 = err1.expand(err1)
    nm.add_all_qubit_quantum_error(err1, ["id"])
    nm.add_all_qubit_quantum_error(err2, ["cx"])
    return nm

def gate_overrotation_noise(theta=0.02):
    nm = NoiseModel()
    # simple proxy: depolarising on CX to mimic coherent mis-calibration
    nm.add_all_qubit_quantum_error(depolarizing_error(theta, 2), ["cx"])
    return nm

def full_noise():
    nm = NoiseModel()
    # combine by re-adding each error type directly
    for builder in (amp_damp_noise, phase_damp_noise, depol_noise,
                    gate_overrotation_noise, thermal_relax_noise):
        sub = builder()            # temporary model
        for inst, q_errs in sub._local_quantum_errors.items():
            for qe in q_errs:
                nm.add_quantum_error(qe, inst[0], inst[1])
    # measurement noise
    for q, ro in readout_noise()._local_readout_errors.items():
        nm.add_readout_error(ro, q)
    return nm


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 5.  Noise dictionary
# ╚═══════════════════════════════════════════════════════════════════════════╝
noise_models = {
    "Ideal"                : None,
    "Amplitude & Phase"    : amp_and_phase_noise(),
    "Amplitude Damping"    : amp_damp_noise(),
    "Depolarizing"         : depol_noise(),
    "Gate Error"           : gate_overrotation_noise(),
    "Measurement"          : readout_noise(),
    "Phase Damping"        : phase_damp_noise(),
    "Thermal Relaxation"   : thermal_relax_noise(),
    "Full Noise"           : full_noise(),
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 6.  Run VQE across all noise channels
# ╚═══════════════════════════════════════════════════════════════════════════╝
energies = {}
print("\nGround-state energies")
print("---------------------")
for lbl, nm in noise_models.items():
    energies[lbl] = vqe_energy(AerSimulator(noise_model=nm))
    print(f"{lbl:22s}: {energies[lbl]:+.6f}  Ha")

ideal_E = energies["Ideal"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# 7.  NAPR
# ╚═══════════════════════════════════════════════════════════════════════════╝
ratios = {lbl: (ideal_E / E if lbl != "Ideal" else 1.0)
          for lbl, E in energies.items()}
NAPR = np.mean([r for k, r in ratios.items() if k != "Ideal"])

E_exact = ideal_E          # for tiny H₂, ideal≈exact


# # ╔═══════════════════════════════════════════════════════════════════════════╗
# # 8.  Bar-plot visualisation
# # ╚═══════════════════════════════════════════════════════════════════════════╝
# labels  = list(energies.keys())
# palette = ['gray', 'red', 'blue', 'green', 'purple', 'orange',
#            'brown', 'pink', 'teal']
# colors  = palette[:len(labels)]

# plt.figure(figsize=(10, 4))
# plt.bar(labels, [energies[l] for l in labels], color=colors)
# plt.axhline(E_exact, color='black', ls='--', label='Exact Energy')
# plt.ylabel('VQE Energy (Hartree)')
# plt.title('H₂  VQE Energy under Noise Channels')
# plt.xticks(rotation=25, ha='right')
# plt.legend(); plt.tight_layout(); plt.show()

# plt.figure(figsize=(10, 4))
# plt.bar(labels, [ratios[l] for l in labels], color=colors)
# plt.axhline(NAPR, color='orange', ls='--', label='Avg NAPR')
# plt.ylabel('Noise-Adapted Performance Ratio (NAPR)')
# plt.title('NAPR across Noise Channels  (higher = better)')
# plt.ylim(0, 1.1 * max(ratios.values()))
# plt.xticks(rotation=25, ha='right')
# plt.legend(); plt.tight_layout(); plt.show()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# 8.  Bar-plot visualisation  (narrow bars + colour + hatch patterns)
# ╚═══════════════════════════════════════════════════════════════════════════╝
import itertools

labels  = list(energies.keys())

# palette & hatch lists (extend automatically if you add more channels)
colours = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2',
           '#59a14f', '#edc949', '#af7aa1', '#ff9da7', '#9c755f']
hatches = ['///', '\*', 'xx', '-|', '+/', 'o/', '..', '*', '/.']

bar_kw  = dict(width=0.55, edgecolor='k', linewidth=0.7)

# –––––––––––––– Energy bar chart ––––––––––––––
plt.figure(figsize=(30, 15))
plt.gca().spines['bottom'].set_linewidth(4)  # X-axis
plt.gca().spines['left'].set_linewidth(4)
plt.gca().spines['right'].set_linewidth(4)
plt.gca().spines['top'].set_linewidth(4)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
for i, (lbl, col, hatch) in enumerate(zip(labels, colours, hatches)):
    plt.bar(i, energies[lbl], color=col, hatch=hatch, **bar_kw)


plt.axhline(E_exact, color='black', ls='--', label='Exact Energy', alpha=0.9, linewidth=5)
plt.ylabel('VQE Energy', fontsize=70)
# plt.title('H₂ VQE Energy across Noise Channels', fontsize=13)
plt.xticks(range(len(labels)), labels, rotation=30, ha='right')
plt.grid(axis='y', alpha=0.25, linestyle=':')
plt.legend(ncol=1, prop={'size': 60}, frameon=True)
plt.tight_layout()
plt.savefig('napr_plots/' + str("VQE_Energy_Metric") + '.png', dpi=100, bbox_inches='tight')
plt.show()

# –––––––––––––– NAPR bar chart ––––––––––––––
plt.figure(figsize=(30, 16))
plt.gca().spines['bottom'].set_linewidth(4)  # X-axis
plt.gca().spines['left'].set_linewidth(4)
plt.gca().spines['right'].set_linewidth(4)
plt.gca().spines['top'].set_linewidth(4)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
for i, (lbl, col, hatch) in enumerate(zip(labels, colours, hatches)):
    plt.bar(i, ratios[lbl], color=col, hatch=hatch, **bar_kw)

plt.axhline(NAPR, color='green', ls='--', label='Avg NAPR', alpha=0.9, linewidth=5)
plt.ylabel('NAPR (> better)', fontsize=80)
# plt.title('NAPR across Noise Channels', fontsize=13)
plt.ylim(0, 1.1 * max(ratios.values()))
plt.xticks(range(len(labels)), labels, rotation=30, ha='right')
plt.grid(axis='y', alpha=0.25, linestyle=':')
plt.legend(ncol=1, prop={'size': 60}, frameon=True)
plt.tight_layout()
plt.savefig('napr_plots/' + str("noise_metric") + '.png', dpi=200, bbox_inches='tight')
plt.show()

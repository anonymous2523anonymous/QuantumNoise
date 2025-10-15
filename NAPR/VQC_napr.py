#!/usr/bin/env python3
# VQC_napr.py  — VQC XOR classification NAPR with expanded noise set and paper-style plotting

import os
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimator   # V1 Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    ReadoutError,
    pauli_error,
)

from scipy.optimize import minimize


def xor_dataset():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,1,1,0], dtype=float)
    return X, y, X.copy(), y.copy()


def circuit(theta, x):
    qc = QuantumCircuit(2)
    qc.ry(np.pi * x[0], 0)
    qc.ry(np.pi * x[1], 1)
    qc.cx(0, 1)
    qc.rz(theta[0], 0); qc.rz(theta[1], 1)
    qc.rx(theta[2], 0); qc.rx(theta[3], 1)
    qc.cx(0, 1)
    qc.rz(theta[4], 0); qc.rz(theta[5], 1)
    return qc


def prob_one(estimator, theta, x):
    qc = circuit(theta, x)
    Z0 = SparsePauliOp(["ZI"], [1.0])
    res = estimator.run([qc], [Z0]).result()
    exp = float(res.values[0])
    p1 = 0.5 * (1.0 - exp)
    return float(np.clip(p1, 1e-6, 1 - 1e-6))


def xent(p, y):
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)))


def train_acc(nm, maxiter=150, shots=4096, seed=11):
    backend = AerSimulator(noise_model=nm, shots=shots)
    est = BackendEstimator(backend)
    Xtr, ytr, Xte, yte = xor_dataset()
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-0.5, 0.5, size=6)

    def loss(th):
        vals = [xent(prob_one(est, th, x), y) for x, y in zip(Xtr, ytr)]
        return float(np.mean(vals))

    res = minimize(loss, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2, "disp": False})
    th_star = res.x
    preds = [(prob_one(est, th_star, x) >= 0.5) for x in Xte]
    acc = float(np.mean((np.array(preds, dtype=int) == yte).astype(float)))
    return acc


def noise_model(kind,
                p1=0.03, p2=0.06,
                gate_p=0.03,
                readout_p=0.03,
                gamma_amp=0.03,
                gamma_phase=0.03,
                T1=100e3, T2=80e3,
                t_1q=50, t_2q=300, t_meas=1000):
    if kind == "ideal":
        return None
    nm = NoiseModel()
    oneq, twoq = ["x", "sx", "rz"], ["cx"]

    def add_amp_phase():
        e1_amp = amplitude_damping_error(gamma_amp)
        e1_phase = phase_damping_error(gamma_phase)
        e1 = e1_phase.compose(e1_amp)
        e2 = e1.tensor(e1)
        for g in oneq: nm.add_all_qubit_quantum_error(e1, g)
        for g in twoq: nm.add_all_qubit_quantum_error(e2, g)

    def add_amp():
        e1 = amplitude_damping_error(gamma_amp)
        e2 = e1.tensor(e1)
        for g in oneq: nm.add_all_qubit_quantum_error(e1, g)
        for g in twoq: nm.add_all_qubit_quantum_error(e2, g)
        e1_amp = amplitude_damping_error(gamma_amp)
        e1_phase = phase_damping_error(gamma_phase)
        e1 = e1_phase.compose(e1_amp)
        e2 = e1.tensor(e1)
        for g in oneq: nm.add_all_qubit_quantum_error(e1, g)
        for g in twoq: nm.add_all_qubit_quantum_error(e2, g)

    def add_depol():
        e1 = depolarizing_error(p1, 1)
        e2 = depolarizing_error(p2, 2)
        for g in oneq: nm.add_all_qubit_quantum_error(e1, g)
        for g in twoq: nm.add_all_qubit_quantum_error(e2, g)

    def add_gate_error():
        e1 = pauli_error([("X", gate_p/3), ("Y", gate_p/3), ("Z", gate_p/3), ("I", 1-gate_p)])
        e2 = pauli_error([("XX", gate_p/9), ("XY", gate_p/9), ("XZ", gate_p/9),
                          ("YX", gate_p/9), ("YY", gate_p/9), ("YZ", gate_p/9),
                          ("ZX", gate_p/9), ("ZY", gate_p/9), ("ZZ", gate_p/9),
                          ("II", 1-gate_p)])
        for g in oneq: nm.add_all_qubit_quantum_error(e1, g)
        for g in twoq: nm.add_all_qubit_quantum_error(e2, g)

    def add_measurement():
        ro = ReadoutError([[1 - readout_p, readout_p], [readout_p, 1 - readout_p]])
        nm.add_all_qubit_readout_error(ro)

    def add_phase():
        e1 = phase_damping_error(gamma_phase)
        e2 = e1.tensor(e1)
        for g in oneq: nm.add_all_qubit_quantum_error(e1, g)
        for g in twoq: nm.add_all_qubit_quantum_error(e2, g)

    def add_thermal():
        e1 = thermal_relaxation_error(T1, T2, t_1q)
        e2 = thermal_relaxation_error(T1, T2, t_2q).tensor(thermal_relaxation_error(T1, T2, t_2q))
        for g in oneq: nm.add_all_qubit_quantum_error(e1, g)
        for g in twoq: nm.add_all_qubit_quantum_error(e2, g)

    if kind == "amplitude_phase":
        add_amp_phase()
    elif kind == "amplitude_damping":
        add_amp()
        add_amp_phase()
    elif kind == "depolarizing":
        add_depol()
    elif kind == "gate_error":
        add_gate_error()
    elif kind == "measurement":
        add_measurement()
    elif kind == "phase_damping":
        add_phase()
    elif kind == "thermal_relaxation":
        add_thermal()
    elif kind == "full":
        add_amp_phase(); add_depol(); add_gate_error(); add_measurement(); add_phase(); add_thermal()
    else:
        raise ValueError(f"Unknown noise kind: {kind}")
    return nm


def napr_from(values, eps=1e-12):
    M0 = values["Ideal"]
    ratios = {k: v/(M0 + eps) for k, v in values.items()}
    arr = np.array([ratios[k] for k in values.keys()])
    NAPR = float(np.mean(arr)); rmin = float(np.min(arr)); var = float(np.mean((arr - NAPR) ** 2))
    return ratios, NAPR, rmin, var


def main():
    outdir = "napr_metric"; os.makedirs(outdir, exist_ok=True)

    labels = [
        "Ideal",
        "Amplitude & Phase",
        "Amplitude Damping",
        "Depolarizing",
        "Gate Error",
        "Measurement",
        "Phase Damping",
        "Thermal Relaxation",
        "Full Noise",
    ]
    kinds = [
        "ideal",
        "amplitude_phase",
        "amplitude_damping",
        "depolarizing",
        "gate_error",
        "measurement",
        "phase_damping",
        "thermal_relaxation",
        "full",
    ]

    values = {}
    for lbl, kind in zip(labels, kinds):
        nm = noise_model(kind)
        acc = train_acc(nm, maxiter=150, shots=4096, seed=11)
        values[lbl] = acc
        print(f"{lbl:>24s}: accuracy = {acc:.3f}")

    ratios, NAPR, rmin, var = napr_from(values)
    print("\nNAPR:", NAPR, " r_min:", rmin, " var:", var)

    colours = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2',
           '#59a14f', '#edc949', '#af7aa1', '#ff9da7', '#9c755f']
    hatches = ['///', '\*', 'xx', '-|', '+/', 'o/', '..', '*', '/.']

    bar_kw  = dict(width=0.55, edgecolor='k', linewidth=0.7)
    plt.figure(figsize=(30, 16))
    ax = plt.gca()
    for spine in ["bottom","left","right","top"]:
        ax.spines[spine].set_linewidth(4)

    for i, (lbl, col, hatch) in enumerate(zip(labels, colours, hatches)):
        plt.bar(i, ratios[lbl], color=col, hatch=hatch, **bar_kw)

    plt.axhline(NAPR, color="green", ls="--", label="Avg NAPR", alpha=0.9, linewidth=5)
    plt.ylabel("NAPR (> better)", fontsize=70)
    # plt.ylim(0, 1.1 * max(ratios.values()))
    plt.ylim(0, 1.75)
    plt.yticks(np.arange(0, 1.75 + 0.001, 0.25), fontsize=70)  # 0, 0.25, 0.50, ..., 1.75
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=60)
    plt.yticks(fontsize=60)
    plt.grid(axis="y", alpha=0.25, linestyle=":")
    plt.legend(ncol=1, prop={"size": 50}, frameon=True)
    plt.tight_layout()
    out_png = os.path.join(outdir, "vqc_napr.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Saved:", out_png)


if __name__ == "__main__":
    main()

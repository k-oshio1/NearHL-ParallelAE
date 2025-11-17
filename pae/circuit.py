import math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate


def make_Ua_ry_qc(n_qubit: int, ry_angle: float):
    """Make operator Ua with Ry gate.

    Args:
        n_qubit (int): number of qubits
        ry_angle (float): ry rotation angle

    Returns:
        QuantumCircuit, float, float: operator Ua, amplitude value, theta value
    """
    if n_qubit != 2:
        raise ValueError("n_qubit must be 2.")
    Ua_qc = QuantumCircuit(n_qubit, name="Ua", global_phase=0)
    Ua_qc.ry(ry_angle, 1)

    amplitude = math.sin(ry_angle/2)**2
    theta = math.asin(math.sqrt(amplitude))

    return Ua_qc, amplitude, theta


def make_V_phi_qc(angle_list: list, Ua_qc: QuantumCircuit):
    """Make operator V_phi from Grover operator with QSP.

    Args:
        angle_list (list): angle sequence
        Ua_qc (QuantumCircuit): operator Ua

    Returns:
        QuantumCircuit: operator V_phi
    """
    n_qubit = Ua_qc.num_qubits
    V_phi_qc = QuantumCircuit(n_qubit + 1, name="V_phi", global_phase=0)
    V_phi_qc_qubit_list = [i for i in range(n_qubit + 1)]

    angle_list_new = []
    for i in range(len(angle_list)-1):
        if i == 0:
            angle_list_new.append(angle_list[0])
            angle_list_new.append(-angle_list[0])
        else:
            angle_list_new.append(angle_list[i] - angle_list_new[-1])
            angle_list_new.append(-angle_list[i] + angle_list_new[-2])

    # print("angle_list     : ", angle_list)
    # print("angle_list_new : ", angle_list_new)
    for ang_index in range(len(angle_list)-1):
        if ang_index == 0:
            V_phi_qc.rx(float(angle_list_new[2*ang_index]) * -2, 0)
            V_phi_qc.rz(math.pi / 2, 0)

            # controlled Grover operator
            # operator Ua
            V_phi_qc.append(Ua_qc, V_phi_qc_qubit_list[1:n_qubit + 1])
            # operator Uf
            V_phi_qc.cz(0, n_qubit)
            # operator Ua†
            V_phi_qc.append(Ua_qc.inverse(), V_phi_qc_qubit_list[1:n_qubit + 1])
            # operator U0
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])
            V_phi_qc.h(n_qubit)
            V_phi_qc.append(MCXGate(n_qubit), V_phi_qc_qubit_list)
            V_phi_qc.h(n_qubit)
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])

            V_phi_qc.rx(float(angle_list_new[2*ang_index + 1]) * -2, 0)
        elif ang_index == len(angle_list)-2: 
            V_phi_qc.rx(float(angle_list_new[2*ang_index] + math.pi/2) * -2, 0)

            # controlled inverse Grover operator
            # operator U0
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])
            V_phi_qc.h(n_qubit)
            V_phi_qc.append(MCXGate(n_qubit), V_phi_qc_qubit_list)
            V_phi_qc.h(n_qubit)
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])
            # operator Ua
            V_phi_qc.append(Ua_qc, V_phi_qc_qubit_list[1:n_qubit + 1])
            # operator Uf
            V_phi_qc.cz(0, n_qubit)
            # operator Ua†
            V_phi_qc.append(Ua_qc.inverse(), V_phi_qc_qubit_list[1:n_qubit + 1])

            V_phi_qc.rz(-math.pi / 2, 0)
            V_phi_qc.rx(float(angle_list_new[2*ang_index + 1] - math.pi/2) * -2, 0)
        elif ang_index % 2 == 0:  # W_φ
            V_phi_qc.rx(float(angle_list_new[2*ang_index]) * -2, 0)
            V_phi_qc.rz(math.pi / 2, 0)

            # controlled Grover operator
            # operator Uf
            V_phi_qc.cz(0, n_qubit)
            # operator Ua†
            V_phi_qc.append(Ua_qc.inverse(), V_phi_qc_qubit_list[1:n_qubit + 1])
            # operator U0
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])
            V_phi_qc.h(n_qubit)
            V_phi_qc.append(MCXGate(n_qubit), V_phi_qc_qubit_list)
            V_phi_qc.h(n_qubit)
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])

            V_phi_qc.rx(float(angle_list_new[2*ang_index + 1]) * -2, 0)
        elif ang_index % 2 != 0:  # W†_φ+π
            V_phi_qc.rx(float(angle_list_new[2*ang_index] + math.pi/2) * -2, 0)

            # controlled inverse Grover operator
            # operator U0
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])
            V_phi_qc.h(n_qubit)
            V_phi_qc.append(MCXGate(n_qubit), V_phi_qc_qubit_list)
            V_phi_qc.h(n_qubit)
            V_phi_qc.x(V_phi_qc_qubit_list[1:n_qubit + 1])
            # operator Ua
            V_phi_qc.append(Ua_qc, V_phi_qc_qubit_list[1:n_qubit + 1])
            # operator Uf
            V_phi_qc.cz(0, n_qubit)

            V_phi_qc.rz(-math.pi / 2, 0)
            V_phi_qc.rx(float(angle_list_new[2*ang_index + 1] - math.pi/2) * -2, 0)

    return V_phi_qc


def make_GHZ_qc(n_qubit:int):
    """Make GHZ state of n-qubits.

    Args:
        n_qubit (int): number of qubits

    Returns:
        QuantumCircuit: quantum circuit to generate GHZ state
    """    
    GHZ_qc = QuantumCircuit(n_qubit, name="GHZ")
    GHZ_qc.h(0)
    l = int(np.ceil(np.log2(n_qubit)))
    for i in range(l, 0, -1):
        for j in range(0, n_qubit, 2 ** i):
            if j + 2 ** (i - 1) >= n_qubit:
                continue
            GHZ_qc.cx(j, j + 2 ** (i - 1))

    return GHZ_qc


def make_parallel_qc(V_phi_qc: QuantumCircuit, n_parallel: int, n_series: int, measurement: str):
    """Make quantum circuit to apply V_ph to GHZ state.

    Args:
        V_phi_qc (QuantumCircuit): operator made by QSP
        n_parallel (int): degree of parallelism
        n_series (int): number of V_ph in series
        measurement (str): measurement type, "X" or "Y" correspond to |±_P> and |±i_P>, respectively

    Returns:
        QuantumCircuit: quantum circuit to apply V_ph to GHZ state
    """    
    n_qubit = V_phi_qc.num_qubits
    n_total_qubits = n_qubit * n_parallel
    parallel_qc = QuantumCircuit(n_total_qubits)
    qubit_list_system = [i for i in range(n_total_qubits)]

    # make GHZ state
    GHZ_qc = make_GHZ_qc(n_parallel)
    parallel_qc.append(GHZ_qc, [i * n_qubit for i in range(n_parallel)])

    for parallel_sequence in range(n_parallel):
        qubit_list_subsystem = qubit_list_system[n_qubit * 
                                                 parallel_sequence:n_qubit*(parallel_sequence+1)]
        for series_i in range(n_series):
            parallel_qc.append(V_phi_qc.decompose().decompose(),
                            qubit_list_subsystem)
    if measurement == "X":
        for i in range(n_parallel):
            parallel_qc.h(n_qubit * i)
    elif measurement == "Y":
        parallel_qc.rz(math.pi / 2, 0)
        for i in range(n_parallel):
            parallel_qc.h(n_qubit * i)

    return parallel_qc
    
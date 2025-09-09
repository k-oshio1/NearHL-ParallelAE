from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def sampling_circuit(circuit: QuantumCircuit, n_shots: int, simulator: AerSimulator):
    """Execute quantum circuit and sampling results.

    Args:
        circuit (QuantumCircuit): quantum circuits to be run
        n_shots (int): number of shots(number of sampling)
        simulator (Aersimulator): circuit simulator

    Returns:
        dict: sampling results
    """
    circuit.measure_all()
    tcirc = transpile(circuit, basis_gates=['u3', 'cx'])
    result = simulator.run(tcirc, shots=n_shots).result()
    sampling_results = result.get_counts(0)

    return sampling_results


def process_result(n_qubit: int, results: dict):
    """Process sampling results to construct the results of entangled measurement.

    Args:
        n_qubit (int): number of qubits
        results (dict): sampling results from Qiskit Sampler

    Returns:
        int: result counts result_0, result_1
    """
    result_0 = 0
    result_1 = 0
    for bit_str, value in results.items():
        temp_bit = bit_str[::-1][0::(n_qubit + 1)]
        if temp_bit.count("1") % 2 == 0:
            result_0 += value
        else:
            result_1 += value

    return result_0, result_1


def calc_oracle_calls(K: int, shots_list: list, n_multiple_list: list, degree_L_X_list: list, degree_L_Y_list: list):
    """Calculates the total number of oracle calls.

    Args:
        K (int): upper limit of the sum in Fisher information 
        shots_list (list): list of numbers of shots
        n_multiple_list (list): list of multiple (degree of parallelism * number of V_ph in series)
        degree_L_X_list (int): approximation degree of QSP (number of oracle in V_ph) for |+_P> and |+i_P> basis measurement
        degree_L_Y_list (int): approximation degree of QSP (number of oracle in V_ph) for |-_P> and |-i_P> basis measurement

    Returns:
        int: total number of oracle calls
    """
    n_orac = 0
    for k in range(K + 1):
        Nk = shots_list[k]
        mk = n_multiple_list[k]
        # NOTE: W_Q contains 1 oracle
        n_orac += Nk * mk * (degree_L_X_list[k] + degree_L_Y_list[k])
    return n_orac

import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from qiskit_aer import AerSimulator

from pae.circuit import make_Ua_ry_qc, make_V_phi_qc, make_parallel_qc
from pae.qsp import calculate_angle_sequence
from pae.estimate import estimate_a_rqpe
from pae.utils import sampling_circuit, process_result, calc_oracle_calls


if __name__ == '__main__':
    # set parameters
    n_qubit = 2
    ry_angle = 0 #1 / 4 * math.pi  # a = sin^2(θ/2)
    K = 9  # 2**(K-1) is maximum number of multiples
    n_trial = 100
    n_parallel_list = [2**k for k in range(K)]
    n_series_list = [1 for _ in range(K)]
    gamma = 4.0835  # parameter derived in Belliardo2020
    nu_K_list = [7.0, 18.0]  # parameter derived in Belliardo2020
    degree_L_X_list = [10, 12, 12, 16, 16, 18, 20, 20, 22] # determined from the preliminary experiment
    degree_L_Y_list = [12, 14, 14, 14, 16, 18, 20, 20, 22] # determined from the preliminary experiment

    # set Qiskit simulator
    simulator = AerSimulator(method='matrix_product_state')

    # make circuits
    Ua_qc, amplitude, theta = make_Ua_ry_qc(n_qubit, ry_angle)

    # print settings
    print("------------------------settings------------------------")
    print(f"n_qubit            : {n_qubit}")
    print(f"n_trial            : {n_trial}")
    print(f"K                  : {K}")
    print(f"max num of multiple: {2**(K-1)}")
    print(f"amplitude(sin^2θ)  : {amplitude}")
    print(f"θ                  : {theta}")
    print(f"n_parallel_list    : {n_parallel_list}")
    print(f"n_series_list      : {n_series_list}")
    print(f"γ                  : {gamma}")
    print(f"ν_K list           : {nu_K_list}")
    print(f"degree_L_X_list    : {degree_L_X_list}")
    print(f"degree_L_Y_list    : {degree_L_Y_list}")
    print("\n")

    # make circuit
    parallel_qc_X_list = []
    parallel_qc_Y_list = []
    for k in range(K):
        angle_list = calculate_angle_sequence(L=degree_L_X_list[k], n_time_length=1) # n_time_length is set as 1, because large T may destabilize the computation of the QSP
        V_phi_qc = make_V_phi_qc(angle_list, Ua_qc)
        parallel_qc_X = make_parallel_qc(V_phi_qc, n_parallel_list[k], n_series_list[k], "X")
        parallel_qc_X_list.append(parallel_qc_X)
        
        angle_list = calculate_angle_sequence(L=degree_L_Y_list[k], n_time_length=1) # n_time_length is set as 1, because large T may destabilize the computation of the QSP
        V_phi_qc = make_V_phi_qc(angle_list, Ua_qc)
        parallel_qc_Y = make_parallel_qc(V_phi_qc, n_parallel_list[k], n_series_list[k], "Y")
        parallel_qc_Y_list.append(parallel_qc_Y)

    # simulate circuit and estimate a
    error_list = np.zeros(K * int(len(nu_K_list)), dtype=float)
    n_multiple_list = [n_parallel * n_series for n_parallel,
                       n_series in zip(n_parallel_list, n_series_list)]

    def calc_error(_):
        a_est_list = []
        for K_max in range(1, K + 1):
            for nu_K in nu_K_list:
                result_list_X = []
                result_list_Y = []
                temp_shots_list = [round(gamma * (K_max - j) + nu_K)
                                   for j in range(1, K_max + 1)]
                for k in range(K_max):
                    sampling_result_X = sampling_circuit(parallel_qc_X_list[k].copy(), temp_shots_list[k], simulator)
                    sampling_result_Y = sampling_circuit(parallel_qc_Y_list[k].copy(), temp_shots_list[k], simulator)
                    result_0, result_1 = process_result(n_qubit, sampling_result_X)
                    result_p, result_m = process_result(n_qubit, sampling_result_Y)
                    result_list_X.append(result_0)
                    result_list_Y.append(result_p)

                temp_a_est_list = estimate_a_rqpe(
                    result_list_X, result_list_Y, n_multiple_list[:K_max], temp_shots_list)
                a_est_list.append(temp_a_est_list[-1])

        return (np.array(a_est_list) - amplitude)**2

    with ProcessPoolExecutor(max_workers=100) as executor:
        all_results = list(executor.map(calc_error, range(n_trial)))
    squared_errors = np.array(all_results)
    sum_squared = np.sum(squared_errors, axis=0)
    error_list = np.sqrt(sum_squared / (n_trial - 1))

    # make list of shots
    oracle_call_list = []
    for K_max in range(1, K + 1):
        for nu_K in nu_K_list:
            temp_shots_list = [round(gamma * (K_max - j) + nu_K)
                               for j in range(1, K_max + 1)]
            oracle_call_list.append(calc_oracle_calls(
                K_max - 1, temp_shots_list, n_multiple_list, degree_L_X_list, degree_L_Y_list))

    print("------------------------results------------------------")
    print("error_list      : ", error_list)
    print("oracle_call_list: ", oracle_call_list)

    output_array = np.stack(
        [(np.array(oracle_call_list)), (np.array(error_list))], 1)
    np.savetxt(fr"./results/result_a{amplitude:.4f}_K{K}_n-trial{n_trial}.csv",
               output_array, delimiter=',', header='Nq, RMSE')

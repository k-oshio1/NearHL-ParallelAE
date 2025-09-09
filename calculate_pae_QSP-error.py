import math, gc
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from concurrent.futures import ProcessPoolExecutor

from pae.circuit import make_Ua_ry_qc, make_V_phi_qc, make_parallel_qc
from pae.qsp import calculate_angle_sequence
from pae.utils import process_result


if __name__ == '__main__':
    # set parameters
    n_qubit = 2
    n_shots = int(1e5)
    K = 9
    n_parallel_list =  [2**i for i in range(K)] 
    n_series_list = [1 for _ in range(K)] 
    ry_angle_list = [2 * math.asin(math.sqrt(a / 100)) for a in range(101)]
    degree_L_list = [l for l in range(4, 26, 2)]

    # print settings
    print("------------------------settings------------------------")
    print(f"n_qubit        : {n_qubit}")
    print(f"n_shots        : {n_shots}")
    print(f"K              : {K}")
    print(f"n_parallel_list: {n_parallel_list}")
    print(f"n_series_list  : {n_series_list}")
    print(f"ry_angle_list  : {ry_angle_list}")
    print(f"degree_L_list  : {degree_L_list}")

    # simulate circuit and estimate oracle conversion error β
    errro_worst_list_p_0 = []
    errro_worst_list_p_p = []

    def calc_p(params):
        simulator = AerSimulator(method='matrix_product_state')
        simulator.set_options(
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1
        )
        ry_angle, angle_list, n_parallel, n_series = params
        Ua_qc, amplitude, theta = make_Ua_ry_qc(n_qubit, ry_angle)
        V_phi_qc = make_V_phi_qc(angle_list, Ua_qc)

        for measurement in ["X", "Y"]:
            parallel_qc = make_parallel_qc(V_phi_qc, n_parallel, n_series, measurement)   
            parallel_qc.measure_all()
            tcirc = transpile(parallel_qc, basis_gates=['u3', 'cx'], optimization_level=0)
            result = simulator.run(tcirc, shots=n_shots).result()
            counts = result.get_counts(0)  
            result = process_result(n_qubit, counts)
            if measurement == "X":
                p_0 = result[0] / n_shots
            elif measurement == "Y":
                p_p = result[0] / n_shots

            del parallel_qc, tcirc, result, counts
            gc.collect()

        p_0_true = (1 + math.cos(n_parallel * n_series * (2 * math.cos(theta * 2)))) / 2
        p_p_true = (1 + math.sin(n_parallel * n_series * (2 * math.cos(theta * 2)))) / 2
        
        return (abs(p_0 - p_0_true), abs(p_p - p_p_true))

    print("\n------------------------results------------------------")
    for n_multiplex_index in range(K):
        print("\n-----------------------------", flush=True)
        print("n_multiplex_index: ", n_multiplex_index + 1, flush=True)
        n_parallel = n_parallel_list[n_multiplex_index]
        n_series = n_series_list[n_multiplex_index]
        n_multiple_list = [n_parallel_list[i]*n_series_list[i] for i in range(K)]

        L_errro_worst_list_p_0 = []
        L_errro_worst_list_p_p = []

        for degree_L in degree_L_list:
            print("degree L: ", degree_L, flush=True)

            parallel_qc_X_list = []
            parallel_qc_Y_list = []
            angle_list = calculate_angle_sequence(L=degree_L)

            params = list(zip(ry_angle_list, [angle_list for _ in range(100)], [n_parallel for _ in range(100)], [n_series for _ in range(100)]))

            if __name__ == "__main__":
                with ProcessPoolExecutor(max_workers=100) as executor:
                    results = list(executor.map(calc_p, params, chunksize=1))
            
            results = [res for res in results if res != (None, None)]

            p_0_diff, p_p_diff = zip(*results) #NOTE: β_{+,k}, β_{i,k} 
            
            L_errro_worst_list_p_0.append(max(p_0_diff))
            L_errro_worst_list_p_p.append(max(p_p_diff))

            print("errro worst p_0: ", L_errro_worst_list_p_0[-1])
            print("errro worst p_p: ", L_errro_worst_list_p_p[-1])

        errro_worst_list_p_0.append(L_errro_worst_list_p_0)
        errro_worst_list_p_p.append(L_errro_worst_list_p_p)

    output_array = np.stack([(np.array(degree_L_list))] 
                            + [(np.array(temp_list)) for temp_list in errro_worst_list_p_0] 
                            + [(np.array(temp_list)) for temp_list in errro_worst_list_p_p]
                            , 1)
    np.savetxt(fr"./results/results_QSP-error_worst.csv", output_array, delimiter=',', 
            header="degree_L," + ",".join([f'p_0 multiple={i}' for i in n_multiple_list]) + "," + ",".join([f'p_p multiple={i}' for i in [2**i for i in range(K)]]))

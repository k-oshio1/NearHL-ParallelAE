import math, random, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pae.estimate import estimate_a_rqpe
from pae.utils import calc_oracle_calls


def exact_prob_sampling(n_shot, p):
    if p > 1:
        print("Error: p must be p <= 1")
        sys.exit(1)
    return sum(1 for _ in range(n_shot) if random.random() < p)


def make_fullseq_list(amplitude, K, n_trial, gamma, n_multiple_list):
    theta = math.asin(math.sqrt(amplitude))

    error_fullseq_list = np.zeros(K * 2, dtype=float)

    # estimate a n_trial times 
    for temp_trial in range(n_trial):
        print("n_trial=(%d/%d)\r" % ((temp_trial + 1), n_trial), flush=True)
        a_est_list = []
        for M_max in range(1, K + 1):
            for x_K in [7.0, 18.0]:
                result_list_X = []
                result_list_Y = []
                temp_shots_list = [round(gamma * (M_max - j) + x_K) for j in range(1, M_max + 1)]
                print(temp_shots_list)
                for m_index in range(M_max):
                    #print("m_index : ", m_index)
                    #print("degree of parallelism : ", n_parallel_list[m_index])
                    #print("number of shots : ", shots_list[m_index])
                    # execute circuit simulation
                    print("test : ", 1 + math.cos(n_multiple_list[m_index] * (2 * math.cos(theta * 2))) / 2)
                    result_0 = exact_prob_sampling(temp_shots_list[m_index], (1 + math.cos(n_multiple_list[m_index] * (2 * math.cos(theta * 2)))) / 2)
                    result_p = exact_prob_sampling(temp_shots_list[m_index], (1 + math.sin(n_multiple_list[m_index] * (2 * math.cos(theta * 2)))) / 2)
                    result_list_X.append(result_0)
                    result_list_Y.append(result_p)

                print("result_list (X measurement): ", result_list_X)
                print("result_list (Y measurement): ", result_list_Y)
                temp_a_est_list = estimate_a_rqpe(result_list_X, result_list_Y, n_multiple_list[:M_max], temp_shots_list) # NOTE: a_est_list is np.array
                print("a_est_list: ", temp_a_est_list)
                a_est_list.append(temp_a_est_list[-1])
            
        error_fullseq_list += (np.array(a_est_list) - amplitude)**2  # list of estimation errors
    error_fullseq_list = (error_fullseq_list / (n_trial-1))**(1/2)

    # make list of shots
    oracle_call_fullseq_list = []
    temp_Lk_list = [10, 14, 22, 34] + [2 * math.ceil((13.63 + 2.72 * n_multiple) / 2) for n_multiple in n_multiple_list[4:]]
    for M_max in range(1, K + 1):
        for x_K in [7.0, 18.0]:
            temp_shots_list = [round(gamma * (M_max - j) + x_K) for j in range(1, M_max + 1)]
            oracle_call_fullseq_list.append(calc_oracle_calls(M_max - 1, 
                                                              temp_shots_list, 
                                                              [1 for _ in range(M_max)],
                                                              temp_Lk_list,
                                                              temp_Lk_list))

    Ua_depth_fullseq_list = temp_Lk_list

    return error_fullseq_list, oracle_call_fullseq_list, Ua_depth_fullseq_list


if __name__ == '__main__':
    # load results
    data_1 = pd.read_csv("./results/result_a0.1464_K9_n-trial100.csv", header=None, skiprows=1)
    Nq_1, RMSE_1 = data_1.iloc[:, 0].astype(float), data_1.iloc[:, 1].astype(float) 
    data_2 = pd.read_csv("./results/result_a0.0000_K9_n-trial100.csv", header=None, skiprows=1)
    Nq_2, RMSE_2 = data_2.iloc[:, 0].astype(float), data_2.iloc[:, 1].astype(float) 


    # make plot
    ## fitting: eps - query (query complexity)
    def model_eps_query(eps, c):
        return (1/eps) * np.e + c * (1/eps) * np.log(1/eps)

    bounds = ([0], [np.inf])
    popt, pcov = curve_fit(model_eps_query, RMSE_1.values, Nq_1.values, bounds=bounds)
    c_est_eps_query = popt
    c_eps_query = int(c_est_eps_query)

    error_log_list = [1/1.1**i for i in range(0, 150)]
    oracle_call_log_list = []
    for i, error in enumerate(error_log_list):
        oracle_call_log_list.append(math.e/error + c_eps_query/error*math.log(1/error))


    ## fitting: eps - depth
    Ua_depth_Fullpara_list = [12, 14, 14, 14, 16, 18, 20, 20, 22]
    def model_eps_depth(eps, c):
        return c * np.log(1/eps)

    bounds = ([0], [np.inf])
    popt, pcov = curve_fit(model_eps_depth, RMSE_1.values[::2], Ua_depth_Fullpara_list, bounds=bounds)
    c_est_eps_depth = popt
    c_eps_depth = round(float(c_est_eps_depth), 1)

    error_log_list = [1/1.1**i for i in range(0, 150)]
    Ua_depth_log_list = []
    for i, error in enumerate(error_log_list):
        Ua_depth_log_list.append(round(float(c_est_eps_depth), 1) * math.log(1/error))


    # calculate query complexity of full sequential case
    ## set parameters
    K = 9
    n_multiple_list = [2**i for i in range(K)]
    n_trial = 100
    gamma = 4.0835
    ## calculate query complexity
    amplitude = math.sin(math.pi/8)**2 
    error_fullseq_1_list, oracle_call_fullseq_1_list, Ua_depth_fullseq_1_list = make_fullseq_list(amplitude, K, n_trial, gamma, n_multiple_list)
    amplitude = 0.0
    error_fullseq_2_list, oracle_call_fullseq_2_list, Ua_depth_fullseq_2_list = make_fullseq_list(amplitude, K, n_trial, gamma, n_multiple_list)


    # calculate query complexity of HL-QAE(Koizumi2025)
    oracle_call_HLQAE_list = [2**i + 1 for i in range(22)]
    error_HLQAE_list = []
    Ua_depth_HLQAE_list = oracle_call_HLQAE_list

    for i, oracle_call in enumerate(oracle_call_HLQAE_list):
        error_HLQAE_list.append(math.pi/2**(i + 1))


    # plot results
    plt.figure(figsize=(16, 3.9))

    ## query complexity plot
    plt.subplot(1, 2, 1)
    plt.grid(True, zorder = 1)

    color_map_1 = plt.get_cmap("autumn")
    color_map_2 = plt.get_cmap("winter")

    plt.plot(Nq_1, RMSE_1, marker='o', markersize=6, linestyle="-", color=color_map_1(1 / 8), label=r"PAE (Full parallel, $a = \sin^2(\pi/8)$)", zorder = 4)
    plt.plot(oracle_call_fullseq_1_list, error_fullseq_1_list, marker='s', markersize=6, linestyle="-", color=color_map_2(1 / 8),  label=r"PAE (Full sequential, $a = \sin^2(\pi/8)$)", zorder = 3)
    plt.plot(Nq_2, RMSE_2, marker='o', markersize=6, linestyle="-", color=color_map_1(4 / 8), label=r"PAE (Full parallel, $a = 0$)", zorder = 2)
    plt.plot(oracle_call_fullseq_2_list, error_fullseq_2_list, marker='s', markersize=6, linestyle="-", color=color_map_2(4 / 8),  label=r"PAE (Full sequential, $a = 0$)", zorder = 2)
    plt.plot(oracle_call_log_list, error_log_list, linestyle="--", color="dimgrey", label=r"$N = \varepsilon^{-1}(e  +$" + f"{c_eps_query}" + r"$\log (1/\varepsilon))$", zorder = 2)
    plt.plot(oracle_call_HLQAE_list, error_HLQAE_list, color="black", label=r"HL-QAE", zorder = 2)
    plt.xlim(1e2, 2e6)
    plt.ylim(1e-4, 1.5e-1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.xlabel(r"$N$ (Total number of $U_a$ and $U_a^\dagger$)", fontsize=16)
    plt.yscale('log')
    plt.ylabel(r"$\varepsilon$ (RMSE)", fontsize=16)
    plt.legend(fontsize=9)
    plt.text(7e4, 1.3e-4, r'$\mathcal{O}(\log(1/\varepsilon))-$depth', rotation=-37, fontsize=14, color='red', ha='left', va='bottom')
    plt.text(2.2e4, 1.0e-4, r'$\mathcal{O}(1/\varepsilon)-$depth', rotation=-39, fontsize=14, color='blue', ha='left', va='bottom')
    plt.text(2.4e3, 1e-4, r'$\mathcal{O}(1/\varepsilon)-$depth', rotation=-39, fontsize=14, color='black', ha='left', va='bottom')

    ## depth plot (plot of a = \sin^2(\pi/8) and \nu_K = 7)
    plt.subplot(1, 2, 2)
    plt.grid(True, zorder = 1 )
    plt.plot(Ua_depth_Fullpara_list, RMSE_1.values[::2], marker='o', markersize=6, linestyle="-", color=color_map_1(1 / 8), label=r"PAE (Full parallel, $a = \sin^2(\pi/8)$)", zorder = 3)
    plt.plot(Ua_depth_fullseq_1_list, error_fullseq_1_list[::2], marker='s', markersize=6, linestyle="-", color=color_map_2(1 / 8),  label=r"PAE (Full sequential, $a = \sin^2(\pi/8)$)", zorder = 3)
    plt.plot(Ua_depth_HLQAE_list, error_HLQAE_list, color="black", label=r"HL-QAE", zorder = 2)
    plt.plot(Ua_depth_log_list, error_log_list, linestyle="--", color="dimgrey", label=r"circuit depth $ =$" + f"{c_eps_depth}" + r"$\log(1/\varepsilon))$", zorder = 2)
    plt.text(1.7e1, 2.5e-4, r'$\mathcal{O}(\log(1/\varepsilon))-$depth', rotation=-80, fontsize=14, color='red', ha='left', va='bottom')
    plt.text(1.9e2, 3e-4, r'$\mathcal{O}(1/\varepsilon)-$depth', rotation=-37, fontsize=14, color='blue', ha='left', va='bottom')
    plt.text(1.3e3, 3e-4, r'$\mathcal{O}(1/\varepsilon)-$depth', rotation=-36, fontsize=14, color='black', ha='left', va='bottom')
    plt.xlim(5, 1e4)
    plt.ylim(2e-4, 1.5e-1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.xlabel(r"circuit depth", fontsize=16)
    plt.yscale('log')
    plt.ylabel(r"$\varepsilon$ (RMSE)", fontsize=16)
    plt.legend(fontsize=9)
    plt.text(1.5e-4, 0.1, "(a)", fontsize=14)
    plt.text(1.4, 0.1, "(b)", fontsize=14)
    plt.subplots_adjust(wspace=0.21) 

    plt.savefig(fr"./results/graph_query_comp.pdf", dpi=600, bbox_inches='tight')
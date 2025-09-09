import math, random, sys
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from pae.estimate import estimate_a_rqpe
from pae.utils import calc_oracle_calls


def exact_prob_sampling(n_shot, p):
    if p > 1:
        print("Error: p must be p <= 1")
        sys.exit(1)
    return sum(1 for _ in range(n_shot) if random.random() < p)


def make_error_list(amplitude):
    theta = math.asin(amplitude**0.5)
    error_list = np.zeros(M * 2, dtype=float)
    for _ in range(n_trial):
        a_est_list = []
        for M_max in range(1, M + 1):
            for x_K in [7.0, 18.0]:
                result_list_X = []
                result_list_Y = []
                temp_shots_list = [round(gamma * (M_max - j) + x_K) for j in range(1, M_max + 1)]
                for m_index in range(M_max):
                    result_0 = exact_prob_sampling(temp_shots_list[m_index], (1 + math.cos(n_multiple_list[m_index] * (2 * math.cos(theta * 2)))) / 2 - beta)
                    result_p = exact_prob_sampling(temp_shots_list[m_index], (1 + math.sin(n_multiple_list[m_index] * (2 * math.cos(theta * 2)))) / 2 - beta)
                    result_list_X.append(result_0)
                    result_list_Y.append(result_p)

                temp_a_est_list = estimate_a_rqpe(result_list_X, result_list_Y, n_multiple_list[:M_max], temp_shots_list) # NOTE: a_est_list is np.array

                a_est_list.append(temp_a_est_list[-1])
        error_list += (np.array(a_est_list) - amplitude)**2  # list of estimation errors
    error_list = (error_list / (n_trial-1))**(1/2)

    return error_list


if __name__ == '__main__':
    # set parameters
    M = 9
    n_trial = 100
    gamma = 4.0835
    n_multiple_list = [2**i for i in range(M)]

    # make list of shots
    oracle_call_exact_oc_list = []
    degree_L_X_list = [1 for _ in range(M)]
    degree_L_Y_list = [1 for _ in range(M)]
    for M_max in range(1, M + 1):
        for x_K in [7.0, 18.0]:
            temp_shots_list = [round(gamma * (M_max - j) + x_K) for j in range(1, M_max + 1)]
            oracle_call_exact_oc_list.append(calc_oracle_calls(M_max - 1, temp_shots_list, n_multiple_list, degree_L_X_list, degree_L_Y_list))

    beta_list = [0.05*i for i in range(6 + 1)]
    error_avg_list = []
    error_max_list = []

    # calculate query complexity
    for i, beta in enumerate(beta_list):
        amplitude_list = [a / 100 for a in range(101)]
        
        with ProcessPoolExecutor(max_workers=100) as executor:
            all_results = list(executor.map(make_error_list, amplitude_list))

        error_list = np.array(all_results)

        error_avg_list.append(np.sum(error_list, axis=0)/len(amplitude_list))
        error_max_list.append(error_list.max(axis=0)) 

    # plot results
    plt.figure(figsize=(16, 6))
    color_map = plt.get_cmap("viridis")

    ## plot average
    plt.subplot(1, 2, 1)
    for i, beta in enumerate(beta_list):
        plt.plot(oracle_call_exact_oc_list, error_avg_list[i], marker='s', linestyle="-", color=color_map(i / len(beta_list)),  label=r"$\beta = $" + f"{beta:.2f}")

    plt.xscale('log')
    plt.xlabel(r"$N$ (Total number of $U_a$ and $U_a^\dagger$)", fontsize=18)
    plt.yscale('log')
    plt.ylim(1e-5, 1)
    plt.ylabel(r"$\varepsilon_{\rm avg}$ (Average RMSE over $a$)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.title("(a)", fontsize=16)
    plt.grid(True)

    ## plot worst
    plt.subplot(1, 2, 2)
    for i, beta in enumerate(beta_list):
        plt.plot(oracle_call_exact_oc_list, error_max_list[i], marker='s', linestyle="-", color=color_map(i / len(beta_list)),  label=r"$\beta = $" + f"{beta:.2f}")
            
    plt.xscale('log')
    plt.xlabel(r"$N$ (Total number of $U_a$ and $U_a^\dagger$)", fontsize=18)
    plt.yscale('log')
    plt.ylim(1e-5, 1)
    plt.ylabel(r"$\varepsilon_{\rm max}$ (Maximum RMSE over $a$)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.title("(b)", fontsize=16)
    plt.grid(True)

    plt.subplots_adjust(wspace=0.3) 

    plt.savefig(fr"./results/graph_query_comp_beta.pdf", dpi=600, bbox_inches='tight')

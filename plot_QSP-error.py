import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # set parameters
    K = 9

    # load results
    data_bp = pd.read_csv(fr"./results/results_QSP-error_worst.csv").iloc[:, 0:(K+1)]
    data_bpi = pd.read_csv(fr"./results/results_QSP-error_worst.csv").iloc[:, [0] + list(range((K+1), (K+1)*2 - 1))]

    # plot
    plt.figure(figsize=(16, 6))
    color_map = plt.get_cmap("viridis")

    ## for β_+
    plt.subplot(1, 2, 1)
    plt.grid(True, zorder=0)

    x = data_bp["# degree_L"][0:]
    lines = data_bp.drop(columns=["# degree_L"]) 

    label_x_p_list = []

    for i, col in list(enumerate(lines.columns)):
        y = data_bp[col][0:]
        plt.plot(x, y, color=color_map(i / K), zorder=3, label=rf"$k={i+1}$") 
        label_x = x[(y[y < 0.05]).first_valid_index()] 
        label_x_p_list.append(int(label_x))
        plt.scatter(label_x, y[(y[y < 0.05]).first_valid_index()] , color=color_map(i / K), marker='o', s=30, zorder=2)
        
    plt.yscale("log")
    plt.xlim(2, 24)
    plt.ylim(6e-3, 1)
    plt.xticks([2 * i for i in range(2, 13)])
    plt.xlabel(r"$L$", fontsize=18)
    plt.ylabel(r"$|\beta_{+}|$", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("(a)", fontsize=16)
    plt.legend(fontsize=12)

    ## for β_+i
    plt.subplot(1, 2, 2)
    plt.grid(True, zorder=0)

    x = data_bpi["# degree_L"][0:]
    lines = data_bpi.drop(columns=["# degree_L"]) 

    label_x_pi_list = []

    for i, col in list(enumerate(lines.columns)):
        y = data_bpi[col][0:]
        plt.plot(x, y, color=color_map(i / K), zorder=3, label=rf"$k={i+1}$") 
        label_x = x[(y[y < 0.05]).first_valid_index()] 
        label_x_pi_list.append(int(label_x))
        plt.scatter(label_x, y[(y[y < 0.05]).first_valid_index()] , color=color_map(i / K), marker='o', s=30, zorder=2)
    
    print("label_x_p_list : ", label_x_p_list)
    print("label_x_pi_list : ", label_x_pi_list)

    plt.yscale("log")
    plt.xlim(2, 24)
    plt.ylim(6e-3, 1)
    plt.xticks([2 * i for i in range(2, 13)])
    plt.xlabel(r"$L$", fontsize=18)
    plt.ylabel(r"$|\beta_{+i}|$", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("(b)", fontsize=16)
    plt.legend(fontsize=12)

    plt.subplots_adjust(wspace=0.3) 

    plt.savefig(fr"./results/graph_QSP-error.pdf", dpi=600)
    plt.show()
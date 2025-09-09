import math
import numpy as np


def estimate_a_rqpe(hit_list_0: list, hit_list_p: list, n_multiple_list: list, shots_list: list):
    """Esimate amplitude with robust phase estimation

    Args:
        hit_list_0 (list): measurement results correspond to |+_P>
        hit_list_p (list): measurement results correspond to |+i_P>
        n_multiple_list (list): list of the number of multiplications
        shots_list (list): list of numbers of shots

    Returns:
        list: list of estimated amplitude
    """    
    small = 1.e-15  # small valued parameter to avoid zero division
    K = len(n_multiple_list)
    a_est_list = np.zeros(K)
    phi_temp = 0.0
    for k in range(K):
        multiple = n_multiple_list[k]
        # estimate Mφ with measurement results
        prob_0_temp = hit_list_0[k] / shots_list[k]
        prob_p_temp = hit_list_p[k] / shots_list[k]
        sine_temp = 2 * prob_p_temp - 1
        cosine_temp = 2 * prob_0_temp - 1 + small
        mxi_temp = math.atan2(sine_temp, cosine_temp) % (2 * math.pi) # wrap in [0, 2π)
        xi_temp = mxi_temp / multiple

        eta = math.floor(phi_temp % (2 * math.pi) / (math.pi / 2**(k - 1)))
        if phi_temp % (2 * math.pi) - xi_temp - (eta - 1) * (math.pi / 2**(k - 1)) <= math.pi / 2**k:
            phi_temp = xi_temp + (eta - 1) * (math.pi / 2**(k - 1)) # in [0, 2π)
        elif xi_temp + (eta + 1) * (math.pi / 2**(k - 1)) - phi_temp % (2 * math.pi) < math.pi / 2**k:
            phi_temp = xi_temp + (eta + 1) * (math.pi / 2**(k - 1)) # in [0, 2π)
        else:
            phi_temp = xi_temp + eta * (math.pi / 2**(k - 1)) # in [0, 2π)
        
        phi_temp = (phi_temp + math.pi) % (2 * math.pi) - math.pi # wrap in [-π, π)

        # calculate amplitude
        a_temp = 0.5 - 0.25 * phi_temp
        a_est_list[k] = a_temp

    return a_est_list

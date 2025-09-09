# NOTE: This file (qsp.py) derives from https://github.com/alibaba-edu/angle-sequence (c) 2020 alibaba-edu,
#       with local modifications by Mizuho Research & Technologies, Ltd. in 2025.
# ------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2020 alibaba-edu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------

import math
import numpy as np
from scipy.special import jn
from pyqsp.angle_sequence import angle_sequence


def calculate_angle_sequence(L=None, n_time_length=1):
    """Calculate angle sequence for Hamiltonian simulation.
    This code is based on https://github.com/alibaba-edu/angle-sequence.

    Args:
        L (int): total term of Laurent polynomial. This value must be even.
        n_time_length (int): time length T in V_{\varphi, T}
    
    Returns:
        list: angle sequence
    """
    if L%2 != 0:
        raise ValueError("L must be even.")
    eps = math.exp(-L / 2 + math.e / 2)
    
    suc = 1 - eps

    a = jn(np.arange(-L / 2, L / 2 + 1, 1), n_time_length)

    angle_list = angle_sequence(a, .9 * eps, suc)
    # print("degree L: ", len(angle_list), flush=True)
    print("angle_list  : ", angle_list, flush=True)

    return angle_list
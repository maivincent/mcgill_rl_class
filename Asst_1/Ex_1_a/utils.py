import numpy as np
import math




def U_function(t, delta, epsilon):
    first_term = 1 + math.sqrt(epsilon)

    inside_log = (math.log(1+epsilon, 2) * t / delta)
    second_term = math.sqrt((1+epsilon) * t * math.log(inside_log, 2) / (2 * t) )

    return first_term * 


def C_function(k, delta, n, epsilon):
    return 2 * U_function(k, delta/n, epsilon)
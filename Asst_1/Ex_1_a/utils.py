import numpy as np
import math




def UFunction(t, delta, epsilon):
    print("t : " + str(t))
    first_term = 1 + math.sqrt(epsilon)

    inside_log = (math.log(1+epsilon * t, 2) / delta)
    print("inside_log : " + str(inside_log))
    second_term = math.sqrt((1+epsilon) * t * math.log(inside_log, 2) / (2 * t) )
    print("2nd term : " + str(second_term))

    return first_term * second_term


def CFunction(k, delta, n, epsilon):
    return 2 * UFunction(k, delta/n, epsilon)
import numpy as np
import math


def draw_softmax(list_action, list_q, temperature):
    assert len(list_action) == len(list_q)
    list_proba = np.zeros(len(list_q))
    for i in range(len(list_action)):
        q = list_q[i] - max(list_q)
        list_proba[i] = math.exp(q / temperature)   # compute each term
    list_proba = list_proba/(np.sum(list_proba))    # normalize
    
    chosen_index = np.random.choice(len(list_action), 1, p=list_proba)
    return list_action[chosen_index[0]]

def getSoftmaxProbas(list_action, list_q, temperature):
    assert len(list_action) == len(list_q)
    list_proba = np.zeros(len(list_q))
    for i in range(len(list_action)):
        q = list_q[i] - max(list_q)
        list_proba[i] = math.exp(q / temperature)   # compute each term
    list_proba = list_proba/(np.sum(list_proba))    # normalize
    return list_proba
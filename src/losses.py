import numpy as np 

def mse(y_true,y_pred):
    return np.mean((y_pred-y_true)**2)
def cross_entropy(y_true, logits):
    max_logits = np.max(logits, axis=1, keepdims=True)

    shifted_logits = logits - max_logits

    log_sum_exp = np.log(
        np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    ) + max_logits

    loss = -np.sum(y_true * logits, axis=1, keepdims=True) + log_sum_exp

    return np.mean(loss)
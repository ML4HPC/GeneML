import numpy as np
from sklearn.metrics import f1_score

def tune_threshold(y_true, y_proba, n_steps=101):

    assert np.all((y_proba >= 0) & (y_proba <= 1))
    assert set(np.unique(y_true)) <= {0,1}

    thresholds = np.linspace(0.001, 0.999, n_steps)
    best_f1 = -np.inf
    best_threshold = thresholds[0]

    for threshold in thresholds:

        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
    return best_threshold

import numpy as np
from sklearn.metrics import roc_auc_score as auc


def roc_auc(y_true, y_score):
    uny = np.unique(y_score)
    aucmean = []
    for i in xrange(0, len(uny)):
        y_trueU = np.copy(y_true)
        y_scoreU = np.copy(y_score)
        np.equal(uny[i], y_score, y_scoreU)
        np.equal(uny[i], y_true, y_trueU)
        if sum(y_trueU) == 0 and sum(y_scoreU) == 0:
            aucmean.append(1)
            continue
        if sum(y_trueU) == 0:
            y_trueU[0] = 1
            y_scoreU[0] = np.mod(y_scoreU[0] + 1, 2)
        if sum(y_trueU) == len(y_trueU) and sum(y_scoreU) == len(y_scoreU):
            aucmean.append(1)
            continue
        if sum(y_trueU) == len(y_trueU):
            y_trueU[0] = 0
            y_scoreU[0] = np.mod(y_scoreU[0] + 1, 2)
        roc_aucU = auc(y_trueU, y_scoreU)
        aucmean.append(roc_aucU)
    return np.mean(aucmean)

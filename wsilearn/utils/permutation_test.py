from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import permutation_test_score
from tqdm import tqdm

def is_ndarray(img):
    return isinstance(img, np.ndarray)

def permutation_test(v1, v2, fct, seed=1, permutations=2000, verbose=False):
    if not is_ndarray(v1):
        v1 = np.array(v1)
    if not is_ndarray(v2):
        v2 = np.array(v2)

    score = fct(v1, v2)
    rs = np.random.RandomState(seed)

    length = len(v1)
    scores = np.zeros(permutations)
    for i in tqdm(range(permutations)):
        # v1_perm = rs.permutation(v1)
        # perm_score = fct(v1_perm, v2)
        perm_score = fct(v1[rs.permutation(length)], v2)
        scores[i] = perm_score

    p = (np.sum(scores >= score) + 1.0) / (permutations + 1)

    print('permutation testing with %d samples and %d permutations: p=%.4f' % (len(v1), permutations, p))
    return p

def roc_auc_permutation_test(labels, probs, seed=1, permutations=2000):
    if not is_ndarray(labels):
        labels = np.array(labels)
    assert (labels.astype(np.uint8)==labels).all()
    labels = labels.astype(np.uint8)
    p = permutation_test(labels, probs, fct=roc_auc_score, seed=seed, permutations=permutations)
    return p

def _test_roc_permutation_test():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X, y = load_breast_cancer(return_X_y=True)
    clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    print('auc: %.4f' % (auc))

    p = roc_auc_permutation_test(y, probs)
    print('p', p, 'mean auc')

if __name__ == '__main__':
    _test_roc_permutation_test()
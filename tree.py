from numba import jit
import numpy as np
import sys

from sklearn.cross_validation import KFold


class DecisionTree(object):
    def __init__(self, max_depth=None, root=None):
        self.left_child = None
        self.right_child = None
        self.predicate = None
        self.predicate_value = None
        self.max_depth = max_depth
        self.output = None
        self.leaf_number = None
        self.root = root
        self.n_leafs = 0
        self.leafs = []
        if root is None:
            self.root = self

    @staticmethod
    @jit(nopython=True)
    def best_split_jit(x, y, sort_index):
        n = sort_index.shape[1]
        mean = np.mean(y[sort_index[0]])
        smean = np.sum((y[sort_index[0]] - mean) ** 2)
        best_feat = 0
        best_val = x[sort_index[0, 0], 0]
        min_gain = smean

        for i in xrange(x.shape[1]):

            cur_ind = sort_index[i]
            y1 = y[cur_ind]
            x1 = x[cur_ind, i]

            mean_l = 0.0
            mean_r = mean
            smean_l = 0.0
            smean_r = smean

            for j in xrange(n - 1):
                delta = y1[j] - mean_l
                mean_l = mean_l + delta / (j + 1.0)
                delta2 = y1[j] - mean_l
                smean_l = smean_l + delta * delta2

                delta = y1[j] - mean_r
                mean_r = mean_r - delta / (n - j - 1.0)
                delta2 = y1[j] - mean_r
                smean_r = smean_r - delta * delta2

                imp_left = smean_l
                imp_right = smean_r

                gain = imp_left + imp_right
                if gain < min_gain and x1[j] != x1[j + 1]:
                    min_gain = gain
                    best_feat = i
                    best_val = (x1[j] + x1[j + 1]) / 2.0

        return best_feat, best_val

    @staticmethod
    def best_split(x, y, sort_index):
        best_feat, best_val = DecisionTree.best_split_jit(x, y, sort_index)
        ind = sort_index[best_feat][x[sort_index[best_feat], best_feat] < best_val]
        mask = np.zeros_like(sort_index, dtype=bool)
        for i in xrange(x.shape[1]):
            mask[i] = np.isin(sort_index[i], ind, assume_unique=True)

        return sort_index[mask].reshape(x.shape[1], -1), sort_index[~mask].reshape(x.shape[1], -1), best_feat, best_val

    def fit(self, x, y, sort_index=None):
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            print 'x and y should be np.ndarray'
            return

        if sort_index is None:
            sort_index = np.asarray(np.argsort(x, axis=0).T)

        if self.max_depth == 0 or sort_index.shape[1] == 1:
            self.output = np.mean(y[sort_index[0]])
            self.leaf_number = self.root.n_leafs
            self.root.n_leafs += 1
            self.root.leafs.append(self)
            return self

        sort_index1, sort_index2, self.predicate, self.predicate_value = self.best_split(x, y, sort_index)

        if sort_index1.shape[1] == 0 or sort_index2.shape[1] == 0:
            self.output = np.mean(y[sort_index[0]])
            self.leaf_number = self.root.n_leafs
            self.root.n_leafs += 1
            self.root.leafs.append(self)
            return self

        if self.max_depth is not None:
            self.left_child = DecisionTree(max_depth=self.max_depth - 1, root=self.root).fit(x, y, sort_index1)
            self.right_child = DecisionTree(max_depth=self.max_depth - 1, root=self.root).fit(x, y, sort_index2)
        else:
            self.left_child = DecisionTree(root=self.root).fit(x, y, sort_index1)
            self.right_child = DecisionTree(root=self.root).fit(x, y, sort_index2)

        return self

    def __predict__(self, x):
        if self.output is not None:
            return self.output
        if x[self.predicate] < self.predicate_value:
            return self.left_child.__predict__(x)
        else:
            return self.right_child.__predict__(x)

    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        for i, xx in enumerate(x):
            y_pred[i] = self.__predict__(xx)
        return y_pred

    def get_leaf(self, x):
        if self.leaf_number is not None:
            return self.leaf_number
        if x[self.predicate] < self.predicate_value:
            return self.left_child.get_leaf(x)
        else:
            return self.right_child.get_leaf(x)


class BernoulliConst(object):
    def __init__(self):
        self.base = None

    def fit(self, x, y, *args):
        self.base = 0.5 * (np.log(1 + np.mean(y)) - np.log(1 - np.mean(y)))
        return self

    def predict(self, x, *args):
        return np.ones(x.shape[0]).reshape(-1, 1) * self.base


class GBoosting(object):
    def __init__(self, max_depth=None, n_iters=10, lr=0.1, init=BernoulliConst()):
        self.max_depth = max_depth
        self.n_iters = n_iters
        self.trees = []
        self.lr = lr
        self.init = init

    @staticmethod
    @jit(nopython=True)
    def calc_new_out(y_hi):
        return np.sum(y_hi) / np.sum(np.abs(y_hi) * (2.0 - np.abs(y_hi)))

    def fit(self, x, y):
        self.init.fit(x, y)
        f_m = self.init.predict(x).reshape(-1)

        for i in xrange(self.n_iters):
            sys.stdout.write("\r" + "iter " + str(i))

            y_h = 2.0 * y / (1.0 + np.exp(2.0 * y * f_m))
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(x, y_h)
            self.trees.append(tree)

            leafs_numbers = np.zeros_like(y, dtype=int)
            for j, xx in enumerate(x):
                leafs_numbers[j] = tree.get_leaf(xx)

            for leaf in tree.leafs:
                ind = leafs_numbers == leaf.leaf_number
                y_hi = y_h[ind]
                leaf.output = self.lr * self.calc_new_out(y_hi)

            f_m += tree.predict(x)

        return self

    @staticmethod
    def loss_fn(y, f):
        return np.sum(np.log(1.0 + np.exp(-2.0 * y * f)))

    def predict(self, x):
        y = self.init.predict(x).reshape(-1) + np.sum(tree.predict(x) for tree in self.trees)
        y_prob = 1.0 / (1 + np.exp(-2.0 * y))
        return 2 * np.array(y_prob > 0.5, dtype=int) - 1

    def staged_predict(self, x):
        y = self.init.predict(x).reshape(-1)
        for tree in self.trees:
            y += tree.predict(x)
            y_prob = 1.0 / (1 + np.exp(-2.0 * y))
            yield 2 * np.array(y_prob > 0.5, dtype=int) - 1

    def predict_proba(self, x):
        y = self.init.predict(x).reshape(-1) + np.sum(tree.predict(x) for tree in self.trees)
        y_prob = 1.0 / (1 + np.exp(-2.0 * y))
        return y_prob


class MeanModel(object):
    def fit(self, x, y):
        pass

    @staticmethod
    def predict(x):
        return np.array(np.mean(x, axis=1) > 0.5, dtype=int)


class Stack(object):
    def __init__(self, meta_model=MeanModel(), models=()):
        self.models = models
        self.meta_model = meta_model

    def fit(self, x, y, n_folds=3):
        folds = KFold(x.shape[0], n_folds)
        meta_features = np.zeros((x.shape[0], len(self.models)))

        for base_index, meta_index in folds:
            x_base, y_base = x[base_index], y[base_index]
            x_meta, y_meta = x[meta_index], y[meta_index]

            for i, model in enumerate(self.models):
                model.fit(x_base, y_base)
                meta_features[meta_index, i] = model.predict(x_meta)

        self.meta_model.fit(meta_features, y)
        for model in self.models:
            model.fit(x, y)

        return self

    def predict(self, x):
        meta_features = np.zeros((x.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            meta_features[:, i] = model.predict(x)
        return self.meta_model.predict(meta_features)

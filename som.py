#!/usr/local/bin/python2.7 -u


import numpy as np


class StopTraining(Exception):
    pass


class CompetitionLayer(np.ndarray):
    def __new__(cls, unit_shape, weight_vec_size,
                alpha0=0.2, sigma0=None, ntrain=500,
                seed=None):
        if not isinstance(weight_vec_size, int):
            raise TypeError(
                'weight_vec_size should be int type, not %s' % type(weight_vec_size)
            )

        ushape = np.array(unit_shape)
        if ushape.size != 2:
            raise ValueError(
                'unit_shape should be 2-dimensional, not %d-dimensional' % ushape.size
            )

        if seed is not None:
            np.random.seed(seed)

        #  axis 0          axis 1          axis 2
        # [unit size of y, unit size of x, weight vector size]
        shp = list(unit_shape) + [weight_vec_size]
        obj = np.random.rand(*shp).view(cls)
        obj.unit_shape = ushape
        obj.unit_size = np.multiply.reduce(ushape)
        obj.weight_vector_size = weight_vec_size

        x, y = np.meshgrid(range(ushape[0]), range(ushape[1]))
        obj.unit_index_list = np.hstack((y.flatten()[:, np.newaxis],
                                         x.flatten()[:, np.newaxis]))

        obj.alpha0 = alpha0
        if sigma0 is None:
            obj.sigma = np.mean(ushape) / 2.
        else:
            obj.sigma = sigma0
        obj.ntrain = ntrain
        obj.itrain = 0
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        else:
            self.unit_shape = getattr(obj, 'unit_shape', None)
            self.unit_size = getattr(obj, 'unit_size', None)
            self.weight_vector_size = getattr(obj, 'weight_vector_size', None)
            self.unit_index_list = getattr(obj, 'unit_index_list', None)
            self.alpha0 = getattr(obj, 'alpha0', None)
            self.sigma = getattr(obj, 'sigma', None)
            self.ntrain = getattr(obj, 'ntrain', None)
            self.itrain = getattr(obj, 'itrain', None)
            return

    @property
    def alpha(self):
        return self.alpha0 * (self.ntrain - self.itrain) / self.ntrain

    @property
    def unit1d(self):
        return self.reshape(self.unit_size, self.weight_vector_size)

    def update_sigma(self):
        self.sigma = 1. + (self.sigma - 1.) * (self.ntrain - self.itrain) / self.ntrain
        return

    def best_matching_unit(self, x):
        d = np.linalg.norm(self.unit1d - x, axis=1)
        return np.unravel_index(d.argmin(), self.unit_shape)

    def neightborhood(self, d_bmu):
        return self.alpha * np.exp(-(d_bmu ** 2) / (2 * self.sigma ** 2))

    def update_weight_vector(self, x):
        bmu = self.best_matching_unit(x)
        d_bmu = np.linalg.norm(self.unit_index_list - bmu, axis=1)
        hc = self.neightborhood(d_bmu)
        new = hc[:, np.newaxis] * (x - self.unit1d)
        self += new.reshape(*self.shape)
        return

    def train(self, x):
        if self.itrain > self.ntrain:
            raise StopTraining(
                'overflow the total number of iteration steps %d' % self.ntrain
            )
        
        self.update_weight_vector(x)
        self.update_sigma()
        self.itrain += 1
        return

    def u_matrix(self):
        import numpy as np
        import scipy.signal as signal

        u = np.empty(self.unit_shape, dtype=float)
        sum_unit_points = np.array(
            [[        1., np.sqrt(2),         1.],
             [np.sqrt(2),          0, np.sqrt(2)],
             [        1., np.sqrt(2),         1.]], dtype=float
        )
        # sum_unit_points = np.array(
        #     [[0., 1., 0.],
        #      [1., 1., 1.],
        #      [0., 1., 0.]], dtype=float
        # )
        nsum = signal.convolve2d(
            np.ones(self.unit_shape, dtype=float),
            sum_unit_points,
            mode='same'
        )
        for j in range(self.unit_shape[0]):
            for i in range(self.unit_shape[1]):
                dm = np.linalg.norm(self.unit1d - self[j,i], axis=1)
                _u = signal.convolve2d(
                    dm.reshape(self.unit_shape), sum_unit_points,
                    mode='same'
                ) / nsum
                u[j, i] = _u[j, i]
        return u

    def unit1d_index_by_unit2d_index(self, u2d_i):
        """
        Return index of unit1d (1-dimensional unit) by that of unit2d (2-dimensional unit)
        :param u2d_i: index of unit2d e.g. (1,2)
        :return: index of unit1d
        """
        import numpy as np
        u2d = np.empty(list(self.unit_shape) + [2], dtype=np.int)
        for iy in range(u2d.shape[0]):
            for ix in range(u2d.shape[1]):
                u2d[iy, ix] = [iy, ix]
        u1d = np.reshape(u2d, (self.unit_size, 2))
        return np.where(np.all(u1d == np.array(u2d_i), axis=1))[0][0]


class KMeans(object):
    def __init__(self, layer2d, k):
        """

        :param layer2d: CompetitionLayer.unit1d
        :param k: A fixed number of cluster
        """
        import copy

        if layer2d.ndim != 2:
            raise ValueError(
                'layer2d should be 2-dimensional, not %d-dimensional' % layer2d.ndim
            )
        if not isinstance(k, int):
            raise TypeError(
                'k should be int type, not %s type' % type(k)
            )
        
        self.layer2d = layer2d
        self.itrain = 0

        # i of layer2d will be in k of k_index
        self.k_index = [[] for j in range(k)]
        for i in range(layer2d.shape[0]):
            j = np.random.randint(0, k)
            self.k_index[j].append(i)
            
        self.last_k_index = None
        return

    @property
    def k(self):
        return len(self)

    def __len__(self):
        return len(self.k_index)

    def get_cluster(self, j):
        if isinstance(j, int):
            i = self.k_index[j]
            return np.array(
                self.layer2d[i]
            )
        else:
            raise TypeError(
                'j should be int type, not %s type' % type(j)
            )

    def size_cluster(self, j):
        return len(self.k_index[j])

    def mean_vector(self):
        # self.layer2d.shape[1] is channel number
        m = np.empty([self.k, self.layer2d.shape[1]], dtype=float)
        for j in range(m.shape[0]):
            m[j] = np.mean(self.get_cluster(j), axis=0)
        return m

    def j_of_k_index(self, i):
        for j, kj in enumerate(self.k_index):
            if i in kj:
                return j
        raise IndexError(
            'index %d is not in self.k_index' % i
        )
        
    def train(self):
        import copy

        self.last_k_index = copy.deepcopy(self.k_index)

        mean_vec = self.mean_vector()
        for i, m in enumerate(self.layer2d):
            # norm from each clusters j
            d = np.linalg.norm(mean_vec - m, axis=1)
            # minimum j
            c = d.argmin()
            # j including i
            j = self.j_of_k_index(i)
            # move i into cluster c
            self.k_index[c].append(self.k_index[j].pop(self.k_index[j].index(i)))
        
        self.itrain += 1
        return

    def is_updated(self):
        import numpy as np

        if self.last_k_index is None:
            raise Exception('object is not trained yet!')

        for j, ki in enumerate(self.k_index):
            curr_ki = np.array(ki)
            last_ki = np.array(self.last_k_index[j])
            if curr_ki.size != last_ki.size:
                return True
            if np.any(curr_ki != last_ki):
                return True
        return False

    def degree_of_homogeneity(self, j):
        import numpy as np

        if j < 0 or self.k <= j:
            raise IndexError('j should be between 0 and %d' % self.k)

        M_j = self.mean_vector()[j]
        d = np.linalg.norm(self.layer2d - M_j, axis=1)
        N_j = len(self.k_index[j])
        S_j = np.sum(d) / N_j
        return S_j

    def degree_of_dissimilarity(self, j, k):
        import numpy as np

        M = self.mean_vector()
        M_j = M[j]
        M_k = M[k]
        return np.linalg.norm(M_j - M_k)

    def davis_bouldin_index(self):
        def R(j):
            R_j = None
            S_j = self.degree_of_homogeneity(j)
            for k in range(self.k):
                if k == j:
                    continue
                S_k = self.degree_of_homogeneity(k)
                D_jk = self.degree_of_dissimilarity(j, k)
                R_jk = (S_j + S_k) / D_jk
                if R_j is None:
                    R_j = R_jk
                else:
                    if R_jk > R_j:
                        R_j = R_jk
            assert R_j is not None
            return R_j

        dbi = 0.
        for j in range(self.k):
            dbi += R(j)
        dbi /= float(self.k)

        return dbi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import tempfile
    import os

    # teachers = np.random.rand(10000, 3)
    teachers = np.array(
        [[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]] * 1000
    )

    m = CompetitionLayer((15,15), 3, ntrain=teachers.shape[0])
    plt.imshow(m)
    plt.show()

    for _ in range(3000):
        i = np.random.randint(0, teachers.shape[0])
        m.train(teachers[i])

    plt.imshow(m)
    u = m.u_matrix()
    cb = plt.contour(u)
    plt.clabel(cb)
    plt.show()

    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    try:
        m.save(f.name)
        m2 = CompetitionLayer.load(f.name)
        print m2.__com_layer_additional_atr

    finally:
        os.remove(f.name)

    k = KMeans(m.unit1d, 3)
    while True:
        k.train()
        if not k.is_updated():
            break

    km = k.mean_vector()
    print km
    plt.imshow(km.reshape([km.shape[0], 1, km.shape[1]]))
    plt.show()

    print k.davis_bouldin_index()


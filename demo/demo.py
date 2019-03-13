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
    
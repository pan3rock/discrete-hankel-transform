from dhtcxx import TrimDHT, DiscreteHankelTransform
import numpy as np
import matplotlib.pyplot as plt


def gaussian(k, k0, a, n):
    fk = k**n / (2.0 * a**2) ** (n+1) * np.exp(- (k-k0)**2 / (4 * a**2))
    return fk


def test_nexp():
    order = 0
    nfull = 2**6
    rmax = 90.0

    a = 0.005
    plt.figure()
    for i in range(4):
        trim_dht = TrimDHT(order, nfull, rmax, i)

        ksample = trim_dht.k_sampling()
        g = gaussian(ksample, 0.3, a, 0) + 0.7 * gaussian(ksample, 0.7, a, 0) \
            + 0.5 * gaussian(ksample, 1.0, a, 0)

        # plt.plot()
        gout = trim_dht.perform(g)
        plt.plot(ksample, gout, '-', label='x{:d}'.format(2**i), alpha=0.9)

    plt.plot(ksample, g, 'k--', label='raw', alpha=0.9)
    plt.xlim([0, 1.5])
    plt.xlabel('k')
    plt.title('M = N x ?')
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_rmax():
    order = 0
    nfull = 2**6

    a = 0.005
    plt.figure()
    for rmax in [30, 60, 90]:
        trim_dht = TrimDHT(order, nfull, rmax, 3)

        ksample = trim_dht.k_sampling()
        g = gaussian(ksample, 0.3, a, 0) + 0.7 * gaussian(ksample, 0.7, a, 0) \
            + 0.5 * gaussian(ksample, 1.0, a, 0)

        # plt.plot()
        gout = trim_dht.perform(g)
        gout = gout / np.amax(np.abs(gout))
        plt.plot(ksample, gout, '-',
                 label='{:.0f}'.format(rmax), alpha=0.9)

    g = g / np.amax(np.abs(g))
    plt.plot(ksample, g, 'k--', label='raw', alpha=0.9)
    plt.legend()
    plt.xlim([0, 1.5])
    plt.xlabel('k')
    plt.title(r'$r_{max}$')
    plt.tight_layout()
    plt.show()

from dhtcxx import DiscreteHankelTransform
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, asin, sqrt


def modified_gaussian_r(r, a, n):
    fr = np.exp(-a**2 * r**2) * r**n
    return fr


def modified_gaussian_k(k, a, n):
    fk = k**n / (2.0 * a**2) ** (n+1) * np.exp(- k**2 / (4 * a**2))
    return fk


def sinc_r(r, a):
    fr = np.sin(a * r) / (a * r)
    return fr


def sinc_k(ks, a, n):
    fk = np.zeros_like(ks)
    for i, k in enumerate(ks):
        if k < a:
            fk[i] = cos(np.pi * n / 2) / (a**2 * sqrt(1.0 - k**2 / a**2)) * \
                (k / a) ** n / (1 + sqrt(1 - k**2 / a**2))
        if k > a:
            fk[i] = sin(n * asin(a / k)) / (a**2 * sqrt(k**2 / a**2 - 1))
    return fk


def get_error(f1, f2):
    ret = 20.0 * np.log10(np.abs(f1 - f2) / np.amax(np.abs(f2)))
    return ret


def plot_dht(r_sample, k_sample, fr_in, fk_out, fk_ref, order):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(r_sample, fr_in, 'r-')
    ax[1].plot(k_sample, fk_out, 'r.-', label='dht')
    if fk_ref is not None:
        ax[1].plot(k_sample, fk_ref, 'k-', label='ref')
    titles = ['Space Function with n={:d}'.format(order),
              'Hankel Function with n={:d}'.format(order)]
    ylabels = ['f(r)', r'f($\rho$)']
    xlabels = ['r', r'$\rho$']
    for i in range(2):
        ax[i].set_title(titles[i])
        ax[i].set_xlabel(xlabels[i])
        ax[i].set_ylabel(ylabels[i])
    if fk_ref is not None:
        plt.legend()
    plt.tight_layout()


def plot_err(k_sample, fk_out, fk_ref, order):
    err = get_error(fk_out, fk_ref)
    plt.figure()
    plt.plot(k_sample, err)
    plt.title('error of the DHT with n={:d}'.format(order))
    plt.ylabel('dB')
    plt.tight_layout()


def test_modified_gaussian():
    order = 1
    rmax = 2
    num = 64
    dht = DiscreteHankelTransform(order, rmax, num)
    r_sample = dht.r_sampling()
    k_sample = dht.k_sampling()
    a = 5.0
    fr_in = modified_gaussian_r(r_sample, a, order)
    fk_out = dht.forward(fr_in)
    fk_ref = modified_gaussian_k(k_sample, a, order)

    plot_dht(r_sample, k_sample, fr_in, fk_out, fk_ref, order)
    plot_err(k_sample, fk_out, fk_ref, order)
    plt.show()


def test_sinc():
    order = 1
    rmax = 26.75
    num = 2**8

    dht = DiscreteHankelTransform(order, rmax, num)
    r_sample = dht.r_sampling()
    k_sample = dht.k_sampling()

    a = 5.0
    fr_in = sinc_r(r_sample, a)
    fk_out = dht.forward(fr_in)
    fk_ref = sinc_k(k_sample, a, order)

    plot_dht(r_sample, k_sample, fr_in, fk_out, fk_ref, order)
    plot_err(k_sample, fk_out, fk_ref, order)
    plt.show()


def ones_r(r):
    return np.ones_like(r)


def test_ones():
    order = 0
    dr = 1.
    num = 2**9
    rmax = num * dr

    dht = DiscreteHankelTransform(order, rmax, num)
    r_sample = dht.r_sampling()
    k_sample = dht.k_sampling()

    fr_in = ones_r(r_sample)
    fk_out = dht.forward(fr_in)

    fk_ref = None
    plot_dht(r_sample, k_sample, fr_in, fk_out, fk_ref, order)
    plt.show()


def test_shift():
    order = 0
    rmax = 100
    dr = 0.1
    num = 2**8
    rmax = num * dr

    dht = DiscreteHankelTransform(order, rmax, num)
    r_sample = dht.r_sampling()
    k_sample = dht.k_sampling()

    fr_in = ones_r(r_sample)
    fr_in = np.zeros(num)
    fr_in[:64] = 1.0
    fk_out = dht.forward(fr_in)

    fk_sig = np.zeros(num)
    fk_sig[np.argmin(np.abs(k_sample-10.0))] = 1.0

    plt.figure()
    plt.plot(k_sample, fk_sig/np.amax(np.abs(fk_sig)))

    res = np.zeros(num-1)
    for m in range(num-1):
        fk_shift = dht.shift(fk_sig, m)
        res[m] = np.sum(fk_out[:-1] * fk_shift)

    plt.plot(k_sample[:-1], res/np.amax(np.abs(res)), '.-')
    plt.tight_layout()
    plt.show()

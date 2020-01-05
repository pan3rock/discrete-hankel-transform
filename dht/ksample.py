#!/usr/bin/env python
from dhtcxx import DiscreteHankelTransform
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sampling in k domain')
    parser.add_argument('--rmax',
                        type=float,
                        help='maximum of receiver distance')
    parser.add_argument('--order',
                        type=int,
                        default=0,
                        help='order of bessel function')
    parser.add_argument('--nr',
                        type=int,
                        help='number of sampling')
    args = parser.parse_args()
    rmax = args.rmax
    order = args.order
    nr = args.nr

    dht = DiscreteHankelTransform(order, nr)
    ksample = dht.k_sampling(rmax)
    rsample = dht.r_sampling(rmax)

    np.savetxt('ksample.txt', ksample)
    np.savetxt('rsample.txt', rsample)

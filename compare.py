from __future__ import print_function
import argparse
import numpy as np


def main(args):
    one = np.loadtxt(args.files[0])
    two = np.loadtxt(args.files[1])
    compare_vectors(one, two)


def compare_vectors(one, two):
    print("equiv?", np.array_equiv(one, two))
    print("same shape?", one.shape == two.shape)

    if one.shape != two.shape:
        print("shape:")
        print('', one.shape)
        print('', two.shape)
        return

    print("all close?", np.allclose(one, two))
    if not np.allclose(one, two):
        sub = two - one
        print("avg(diff)", sub.mean())
        print("rmse", np.sqrt(np.mean(sub**2)))
        print("max(abs(diff))", np.abs(sub).max())
        print("histo(diff)", np.histogram(sub))
        #howclose(sub)


def howclose(sub):
    prevnum = -1
    for p in range(1, 100):
        eps = 10**(-p)
        num = len([a for a in sub if np.abs(a) < eps])
        if num == prevnum:
            break
        prevnum = num
        print("how close? eps={:,} {}".format(eps, num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='*', default=["one.txt", "two.txt"], help="vectors to compare")
    main(parser.parse_args())

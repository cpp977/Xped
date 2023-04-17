import h5py
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description="Plot hdf5 file.")
parser.add_argument(type=str, dest="filename", help="file name")
parser.add_argument("-q", type=str, dest="quantity", choices=["grad", "energy", "both"], default="both", help="quantity (either energy or grad)")

args = parser.parse_args()

f = h5py.File(args.filename, 'r')

cost = np.array(f["cost"])
grad = np.array(f["grad_norm"])

f.close()

if args.quantity == "grad":
    plt.plot(grad, 'r+-', label="grad_norm")
    plt.yscale("log")
    plt.ylabel("grad")
    plt.grid()
elif args.quantity == "energy":
    plt.plot(cost, 'r+-', label="energy")
    plt.ylabel("E")
    plt.grid()
else:
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(cost, 'r+-', label="energy")
    ax1.set_ylabel("E")
    ax2.plot(grad, 'b+-', label="grad_norm")
    ax2.set_yscale("log")
    ax2.set_ylabel("grad")
    ax1.grid()
    ax2.grid()
plt.xlabel("iter")
plt.show()

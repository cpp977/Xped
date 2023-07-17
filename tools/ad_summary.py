import h5py
import numpy as np
import matplotlib.pyplot as plt

import argparse

def deduce_Ls(filename):
    parts = filename.split("_")
    Lx = None
    Ly = None
    for part in reversed(parts):
        if "Ly" in part:
            elems = part.split("=")
            Ly = int(elems[-1])
        if "Lx" in part:
            elems = part.split("=")
            Lx = int(elems[-1])
    return Lx, Ly

parser = argparse.ArgumentParser(description="Plot hdf5 file.")
parser.add_argument(nargs="+", type=str, dest="filenames", help="file names")
parser.add_argument("-q", type=str, dest="quantity", choices=["grad", "energy", "both"], default="both", help="quantity (either energy or grad)")
parser.add_argument('-s', action='store_true')

args = parser.parse_args()

for filename in args.filenames:
    f = h5py.File(filename, 'r')
    Lx, Ly = deduce_Ls(filename)
    label="unknown"
    if Lx is not None and Ly is not None:
        label = f"{Lx}x{Ly}"
    cost = np.array(f["cost"])
    grad = np.array(f["grad_norm"])

    f.close()

    if args.quantity == "grad":
        plt.plot(grad, '+-', label=label)
        plt.yscale("log")
        plt.ylabel("grad")
    elif args.quantity == "energy":
        plt.plot(cost, '+-', label=label)
        plt.ylabel("E")
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

plt.legend()
plt.grid()
if args.s:
    plt.savefig(f"{args.quantity}.pdf")
else:
    plt.show()

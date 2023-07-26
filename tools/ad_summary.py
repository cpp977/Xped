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
parser.add_argument("-q", type=str, dest="quantity", choices=["grad", "energy", "energy-rel", "both"], default="both", help="quantity (either energy or grad)")
parser.add_argument("-exact", type=float, dest="E_exact", default=None, help="Exact energy")
parser.add_argument('-s', action='store_true')

args = parser.parse_args()

if args.quantity == "energy-rel" and args.E_exact is None:
    raise ValueError("Specify the exact energy with `-exact <val>` if you want to plot the relative energy.")

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
    elif args.quantity == "energy-rel":
        plt.plot((cost-args.E_exact)/np.abs((cost+args.E_exact)), '+-', label=label)
        plt.ylabel("E")
        plt.yscale("log")
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

if args.quantity == 'energy' and args.E_exact is not None:
    plt.axhline(y=args.E_exact, color='r', linestyle='-', label='exact')
    
plt.legend()
plt.grid()
if args.s:
    plt.savefig(f"{args.quantity}.pdf")
else:
    plt.show()

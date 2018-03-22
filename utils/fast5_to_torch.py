"""
Convert a fast5 file to (med-mad normalized) torch signal indices/data.
"""
import fast5
import torch
import numpy as np
import argparse
import os
from glob import glob


def medmad_scale(arr):
    """Perform med-mad scaling of a numpy array."""
    # edge case:
    if arr.shape[0] == 1: return np.array([0.0], dtype=float)
    # compute medmad-scaled array:
    arr_med = np.median(arr)
    _GAMMA_ = 1.4826
    arr_mad = _GAMMA_ * np.median(np.abs(arr - arr_med))
    arr_medmad = (arr - arr_med) * np.reciprocal(arr_mad)
    # return:
    return arr_medmad


def fast5_to_torch(fast5_glob, out_dir, prefix):
    """
    Read a fast5 file and extract signal; output <PREFIX>.idxs.pth, <PREFIX>.data.pth
    inside `out_dir`.

    Args:
    * fast5_file: path to fast5 file.
    * out_dir: path to directory for output files.
    * prefix: what goes before `.{idxs,data}.pth` in the filename.

    Returns:
    N/A (writes output to file)
    """
    # read fast5 signals into numpy arrays and perform medmad-normalization:
    signals = []
    lengths = []
    ctr = 0
    for f5file in glob(fast5_glob):
        try:
            signal = medmad_scale(np.array([sample for sample in f5file.get_raw_samples()], dtype=np.float32))
            signals.append(signal)
            lengths.append([ctr, ctr+signal.shape[0]])
            ctr += signal.shape[0]
        except:
            raise IOError("No samples found in fast5 file")

    # write to torch tensors:
    concat_signals = np.concatenate(signals, axis=0)
    torch.save(torch.from_numpy(concat_signals).float(), os.path.join(out_dir,"{}.data.pth".format(prefix)))
    torch.save(torch.IntTensor(lengths), os.path.join(out_dir,"{}.idxs.pth".format(prefix)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dump Fast5 files to Pytorch tensors.")
    parser.add_argument("fast5_glob")
    parser.add_argument("--dir", dest="out_dir", default="./", help="Output directory [pwd]")
    parser.add_argument("--prefix", dest="prefix", default="signals", help="Output filename prefix [signals]")
    args = parser.parse_args()
    fast5_to_torch(args.fast5_glob, args.out_dir, args.prefix)

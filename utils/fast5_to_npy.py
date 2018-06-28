"""
Convert a fast5 file to (med-mad normalized) numpy signal indices/data.
"""
import fast5
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


def fast5_to_numpy(fast5_glob, out_dir, prefix):
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
    for f5file in tqdm(glob(fast5_glob)):
        try:
            signal = medmad_scale(np.array([sample for sample in fast5.File(f5file).get_raw_samples()], dtype=np.float32))
            signals.append(signal)
            lengths.append([ctr, ctr+signal.shape[0]])
            ctr += signal.shape[0]
        except:
            raise IOError("No samples found in fast5 file")

    # write to numpy tensors:
    concat_signals = np.concatenate(signals, axis=0)
    np.save(os.path.join(out_dir,"{}.data.npy".format(prefix)), concat_signals)
    np.save(os.path.join(out_dir,"{}.idxs.npy".format(prefix)), np.array(lengths, dtype=int))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dump Fast5 files to Pytorch tensors.")
    parser.add_argument("fast5_glob")
    parser.add_argument("--dir", dest="out_dir", default="./", help="Output directory [pwd]")
    parser.add_argument("--prefix", dest="prefix", default="signals", help="Output filename prefix [signals]")
    args = parser.parse_args()
    fast5_to_numpy(args.fast5_glob, args.out_dir, args.prefix)

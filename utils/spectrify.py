"""
spectrify.py: CLI utility running a gaussian spectral method over all signals in a signal.pth file.

N.B.: this operation is highly memory-intensive; if you have 1000 gaussian components and 1M samples,
this requires 32GB of RAM. This script is best run on an HPC cluster.
"""
import torch
import numpy as np
import os
import argparse

def gaussian_spectrum(seq, means, stdvs, keep_sample=True):
    """
    Take a sequence and apply a gaussian over it.
    
    `seq` expected to be a torch.FloatTensor of shape `(num_samples,)`; returns a spectrum
    FloatTensor of shape `(num_samples, num_gaussians+1)` containing the gaussian likelihoods
    appended to each raw sample.
    
    'means' and 'stdvs' should each be a numpy array of floats of the same length;
    the length is the number of components. We assume that the k-th gaussian is defined
    by the parameters `(means[k], stdvs[k])`.
    """
    # construct gaussian components:
    num_gaussians = means.shape[0]
    gaussians = torch.distributions.Normal(torch.from_numpy(means),torch.from_numpy(stdvs))
    
    # expand input sequence to `(num_samples, num_gaussians)`:
    signals = seq.unsqueeze(1).expand(seq.size(0), num_gaussians)

    # compute probabilities:
    output = gaussians.log_prob(signals).exp()
    
    # return probabilities (possibly with original sample appended):
    if keep_sample: output = torch.cat((seq.unsqueeze(1),output), dim=1)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run gaussian-spectral mapping on signal.")
    parser.add_argument('signal_path', help="Path to input file.")
    parser.add_argument('model_path', help="Path to NPZ file containing gaussian component parameters.")
    parser.add_argument('out_path', help="Path of output file.")
    parser.add_argument('--keep_sample', dest='keep_sample', default=True, help="Keep original sample as extra column [True]")
    args = parser.parse_args()
    assert os.path.exists(args.signal_path), "Signal file does not exist!"
    assert os.path.exists(args.model_path), "Model file does not exist!"
    raw_signal = torch.load(args.signal_path)
    parameters = np.load(args.model_path)
    assert (parameters['means'].shape[0] == parameters['stdvs'].shape[0]), "Means and Stdvs don't match!"
    gauss_signal = gaussian_spectrum(raw_signal, parameters['means'], parameters['stdvs'], keep_sample=args.keep_sample)
    torch.save(gauss_signal, args.out_path)

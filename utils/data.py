"""
Sequential data-loading helper classes.

Usage:

>>> seqdata = SeqTensorDataset(sig_indices, signals, seq_indices, sequences) 
>>> loader = torch.utils.data.DataLoader(seqdata, batch_size=8, shuffle=True, collate_fn=sequence_collate_fn)
>>> for (x, xlen, y, ylen) in loader:
>>>     # [... perform training loop and computations here ...]
"""
import numpy as np
import torch
import torch.utils.data as data
import os


class SeqTensorDataset(data.Dataset):
    """
    Constructor arguments:
    * sig_idx_tensor: of shape (num_examples, 2). The start/stop indices for each signal.
    * signal_tensor: of shape (total_samples, [signal_dim]).
    * seq_idx_tensor: of shape (num_examples, 2). The start/stop indices for each sequence.
    * sequence_tensor: of shape (total_seq_length).
    
    In the above, `total_samples` is equal to the sum of the lengths of all the signals and 
    `total_seq_length` is equal to the sum of the lengths of all the ground truth sequences.
    We enforce a condition that fetched signals must be of shape (signal_length, D) where D may be == 1.
    If the input signal tensor is a 1D sequence of shape (total_samples,), we unsqueeze the final dimension.

    `__getitem__` returns a tuple with the following components:
    * _signal: a FloatTensor of size (signal_length, max{1,signal_dim}).
    * _seq: an IntTensor of size (seq_length,).
    """
    def __init__(self, sig_idx_tensor, signal_tensor, seq_idx_tensor, sequence_tensor):
        # sanity check:
        assert (sig_idx_tensor.size(0) == seq_idx_tensor.size(0)), "Must have same number of examples in all tensors"
        unsqueeze = (len(signal_tensor.size()) == 1) # bool indicating whether to unsqueeze
        # save tensors as object fields:
        self.seq_indices = seq_idx_tensor
        self.sig_indices = sig_idx_tensor
        self.signals = signal_tensor if not unsqueeze else signal_tensor.unsqueeze(-1)
        self.sequences = sequence_tensor

    def __getitem__(self, index):
        # get signal:
        sig_start, sig_stop = self.sig_indices[index]
        _signal = self.signals[sig_start:sig_stop]
        # get sequence:
        seq_start, seq_stop = self.seq_indices[index]
        _seq = self.sequences[seq_start:seq_stop]
        return (_signal, _seq)

    def __len__(self):
        return self.seq_indices.size(0)


class SignalDataset(data.Dataset):
    """
    A signals-only version of SeqTensorDataset, to be used for basecalling.

    Constructor arguments:
    * sig_idx_tensor: of shape (num_examples, 2). The start/stop indices for each signal.
    * signal_tensor: of shape (total_samples, [signal_dim]).
    
    In the above, `total_samples` is equal to the sum of the lengths of all the signals.
    We enforce a condition that fetched signals must be of shape (signal_length, D) where D may be == 1.
    If the input signal tensor is a 1D sequence of shape (total_samples,), we unsqueeze the final dimension.

    `__getitem__` returns `_signal` ~FloatTensor of size (signal_length, max{1,signal_dim}).
    """
    def __init__(self, sig_idx_tensor, signal_tensor):
        unsqueeze = (len(signal_tensor.size()) == 1) # bool indicating whether to unsqueeze
        # save tensors as object fields:
        self.sig_indices = sig_idx_tensor
        self.signals = signal_tensor if not unsqueeze else signal_tensor.unsqueeze(-1)

    def __getitem__(self, index):
        # get signal:
        sig_start, sig_stop = self.sig_indices[index]
        return self.signals[sig_start:sig_stop]

    def __len__(self):
        return self.sig_indices.size(0)


class BatchSignalTransform(object):
    """
    Given a sequential signal tensor (**not** a Variable), split into a batch.
    """
    def __init__(self, max_length=2000, overlap=0, max_batch_size=16, drop_remainder=True):
        """Save input arguments."""
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        assert (overlap == 0), "[BatchSignalTransform] overlap != 0 currently unsupported"
        assert drop_remainder, "[BatchSignalTransform] padding/saving remainder data currently unsupported"

    def __call__(self, signal):
        """Split a signal ~ (signal_length, dim) into a batch ~ (batch_size, min{max_length, len(signal)}, dim)."""
        # squeeze out batch dimension:
        signal = signal.squeeze(0) if (len(signal.size()) == 3) else signal

        # if we don't hit max length, just return a batch of size 1:
        if signal.size(0) < self.max_length:
            return signal.unsqueeze(0)

        # compute number of examples in the batch:
        bsz = signal.size(0) // self.max_length
        bsz = min(bsz, self.max_batch_size) # (clamp to max value)
        remaining = (signal.size(0) % self.max_length > 0)

        # reshape into a batch:
        return signal[0:self.max_length*bsz].view(bsz, self.max_length, signal.size(1))


def pad_sequence(sequences, batch_first=False, pad_value=0):
    """
    Take a bunch of sequences and pad them together. This is essentially the same as the
    implementation of `torch.utils.rnn.pad_sequence` with the main difference being that
    it supports a list of `Tensor`s instead of strictly requiring `Variable`s.

    Args:
    * sequences: a list of tensors, all of which are of shape `(T,[D])`.
    * batch_first: bool. If True, return batched sequences in BxT[xD] format; if False,
    return in TxB[xD] format.
    * pad_value: padding value to place in the batch [0]

    Returns:
    * out_batch: a tensor of shape TxB[xD].

    Credits: half of this function is taken from the implementation of
    `torch.utils.rnn.pad_sequence`.
    """
    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = max([seq.size() for seq in sequences], key=lambda sz: sz[0])
    max_len, trailing_dims = max_size[0], max_size[1:]
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    # construct batch with zero-padding:
    out_batch = torch.zeros(*out_dims).add_(pad_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_batch[i, :length, ...] = seq
        else:
            out_batch[:length, i, ...] = seq

    return out_batch


def sequence_collate_fn(data):
    """
    The `collate_fn` field of data.DataLoader requires a function that determines
    how a list of `[(data_seq, target_seq)]` of length `batch_size` gets merged together
    into a single batched tensor. Here, we perform dynamic padding to the maximum length
    of the sequences in the batch.

    Args:
    * data: list of tuples of tensors of the form `(data_seq, target_seq)`.

    Returns: four tensors of the form `(input_batch, input_lengths, target_batch, target_lengths)`, where
    * input_batch: 3D FloatTensor of shape (batch_size, max_in_seq_length, in_dim).
    * input_lengths: 1D IntTensor of shape (batch_size); the lengths of each input sequence.
    * target_batch: 2D FloatTensor of shape (batch_size, max_target_seq_length).
    * target_lengths: 1D IntTensor of shape (batch_size); the lengths of each target sequence.

    Credits: the `collate_fn` implementation in Yunjey Choi's data loader provided helpful insight.
    (https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py#L39)
    """
    # sort dataset by decreasing signal length:
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # get lengths:
    input_lengths = torch.IntTensor([d[0].size(0) for d in data])
    target_lengths = torch.IntTensor([d[1].size(0) for d in data])

    # batch sequence tensors:
    input_batch = pad_sequence([d[0] for d in data], batch_first=True)
    target_batch = pad_sequence([d[1] for d in data], batch_first=True, pad_value=0)

    return (input_batch, input_lengths, target_batch, target_lengths)


def concat_labels(seq_batch, lengths):
    """
    Convert a batch of padded sequences of discrete labels to a single concatenated sequence.
    
    Args:
    * seq_batch: a 2D tensor of shape (batch_size, max_seq_length).
    * lengths: a 1D IntTensor of sequence lengths.
    
    Returns:
    A single IntTensor of length `sum{len(seq) for seq in seq_batch}`.
    """
    return torch.cat([seq_batch[k,0:lengths[k]] for k in range(lengths.size(0))], dim=0)


def split_labels(merged_seq, lengths):
    """
    Split a sequence at the corresponding lengths and pad into a batch.

    Currently only tested for `merged_seq` of shape `(total_num_labels,)`.
    
    Returns:
    an IntTensor of shape `(num_seqs, max(lengths))`.
    """
    ctr = 0
    maxlen = 0
    seq_batch = []
    for k in range(lengths.size(0)):
        start = ctr
        stop = start+lengths[k]
        seq_batch.append(merged_seq[start:stop])
        ctr += lengths[k]
        if (lengths[k] > maxlen):
            maxlen = lengths[k]
    #torch.zeros(
    # TODO: WORK IN PROGRESS


def mask_padding(seqs, seq_lengths, fill_logit_idx=0):
    """
    Given a batch of logit sequences, clamp everything after each sequence's max length to some value (usually
    the dimension of each logit that represents a padding value).
    
    This is primarily useful for allowing beam decoders to work properly at validation time. (Note that this
    does not leak any information about the target sequence, as all operations here are performed on knowledge
    of the input sequences.)

    Args:
    * seqs: FloatTensor or variable of shape (batch, dim(logits), max(seq_lengths)) ~ (B x T x D). The logit sequences.
    * seq_lengths: IntTensor or variable of shape (batch,). The lengths of each sequence.
    * fill_logit_idx: the logit index to put 100% of the weight upon; this is usually the dimension of each logit
    that represents the <PAD>, <NULL>, or <EMPTY> character.

    Returns:
    * out: FloatTensor or variable of same shape as `seqs`, with all values past `seq_lengths` masked to the specified NULL token.
    """
    if isinstance(seqs, torch.autograd.Variable):
        _seqs = seqs.data
    else:
        _seqs = seqs
    if isinstance(seq_lengths, torch.autograd.Variable):
        _seq_lengths = seq_lengths.data
    else:
        _seq_lengths = seq_lengths

    # construct a tensor which is full of <PAD> values:
    out_tsr = _seqs.new(_seqs.shape)
    out_tsr[:,:,fill_logit_idx] = 1.
    
    # copy over seqs:
    for b in range(out_tsr.size(0)):
        out_tsr[b][0:_seq_lengths[b]] = _seqs[b][0:_seq_lengths[b]]

    if isinstance(seqs, torch.autograd.Variable):
        return torch.autograd.Variable(out_tsr)
    else:
        return out_tsr

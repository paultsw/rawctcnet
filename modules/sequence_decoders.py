"""
sequence_decoders.py: functions to convert a sequence of logits to integer strings.
"""
import torch
import torch.nn.functional as F


def argmax_decode(logits):
    """
    Given a batch of logits represented as a tensor, 

    Args:
    * logits: a FloatTensor or FloatTensor variable of shape (batch, sequence, logit). The final
    coordinate of the `logit`-dimension is assumed to be the probability of the blank label.
    
    Returns:
    * labels: a LongtTensor or LongTensor variable of shape (batch, sequence), where each entry
    is the integer labeling of the logit, based on the argmaxed coordinate.
    """
    labels = logits.new(logits.size(0), logits.size(1))
    _, labels = torch.max(logits, dim=2)
    return labels


def labels2strings(labels, lookup={0: '', 1: 'A', 2: 'G', 3: 'C', 4: 'T'}):
    """
    Given a batch of labels, convert it to string via integer-to-char lookup table.

    Args:
    * labels: a LongTensor or LongTensor variable of shape (batch, sequence).
    * lookup: a dictionary of integer labels to characters. One of the labels should map to the
    empty string to represent the BLANK label.

    Returns:
    * strings_list: a list of decoded strings.
    """
    if isinstance(labels, torch.autograd.Variable): labels = labels.data
    labels_py_list = [list(labels[k]) for k in range(labels.size(0))]
    strings_list = [ "".join([lookup[ix] for ix in labels_py]) for labels_py in labels_py_list ]
    return strings_list
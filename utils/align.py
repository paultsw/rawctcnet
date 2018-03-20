"""
Implementation of local alignment via SSW for validation output logs.
"""
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

_DNAFULL_ = {
    ('A', 'A') : 5.0,
    ('A', 'G') : -4.0,
    ('A', 'C') : -4.0,
    ('A', 'T') : -4.0,
    ('G', 'A') : -4.0,
    ('G', 'G') : 5.0,
    ('G', 'C') : -4.0,
    ('G', 'T') : -4.0,
    ('C', 'A') : -4.0,
    ('C', 'G') : -4.0,
    ('C', 'C') : 5.0,
    ('C', 'T') : -4.0,
    ('T', 'A') : -4.0,
    ('T', 'G') : -4.0,
    ('T', 'C') : -4.0,
    ('T', 'T') : 5.0
}

def ssw(seq1, seq2):
    """
    Take two DNA sequences and return a local alignment, alignment score, and percent-identity.

    Args:
    * seq1, seq2: DNA sequences; strings with characters in [A,G,C,T].

    Returns: a tuple (aln, score, pct_id) where:
    * aln_str: a printable output representing alignments, with a middle line containing {|, . } characters.
    * score: float value representing alignment score.
    """
    aln_data = pairwise2.align.localdd(seq1, seq2, #  sequences A and B
                                       _DNAFULL_, # use scoring matrix as above
                                       -10.0, -0.5, # sequence A gap-open and gap-extend penalties
                                       -10.0, -0.5, # sequence B gap-open and gap-extend penalties
                                       one_alignment_only=True)[0] # only return best alignment
    aln_str = format_alignment(*aln_data)
    score = float(aln_data[2])
    return aln_str, score

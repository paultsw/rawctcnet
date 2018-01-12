"""
Command-line argument parser utilities.
"""
import os

def parse_dataset_paths(datasets_str):
    """
    Take a string of dataset paths and convert them into a list of 3-tuples of dataset paths.
    """
    # basic validation (these asserts won't catch everything that can go wrong)
    num_commas = datasets_str.count(',')
    num_semicolons = datasets_str.count(';')
    assert (num_semicolons == (num_commas / 2 - 1)), 'Dataset path string improperly-formatted'
    assert (num_commas % 2 == 0), 'Dataset path string improperly-formatted'

    # format into paths
    datasets = datasets_str.strip().split(';')
    data_paths = []
    for ds in datasets:
        dsp = tuple(ds.split(','))
        assert (len(dsp) == 3), 'Dataset path string improperly-formatted'
        data_paths.append( dsp )
       
    # do a final path-validation check:
    for x,y,z in data_paths:
        assert os.path.exists(x), "Data file at {0} does not exist".format(x)
        assert os.path.exists(y), "Data file at {0} does not exist".format(y)
        assert os.path.exists(z), "Data file at {0} does not exist".format(z)

    return data_paths

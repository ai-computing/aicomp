from typing import OrderedDict
import h5py
import numpy as np
import argparse
import logging
from tqdm import tqdm
from itertools import repeat, cycle
import json
import glob
import random

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(
    description="Training data sharding for BERT.")
parser.add_argument(
    '--input_hdf5',
    type=str,
    default='hdf5',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5',
    type=str,
    default='',
    help='Output hdf5_dir path')
args = parser.parse_args()


input_files = sorted(glob.glob(args.input_hdf5 + '/part_*.hdf5', recursive=False))
num_shards = len(input_files)
logging.info('n_input_shards = {}'.format(num_shards))

ifile_handles={}
for ifile_idx in tqdm(range(num_shards)):
    handle = h5py.File(f'{input_files[ifile_idx]}', 'r')
    print(handle.keys())
    ifile_handles[ifile_idx] = [
        handle['input_ids'][:],
        handle['input_mask'][:],
        handle['segment_ids'][:],
        handle['masked_lm_positions'][:],
        handle['masked_lm_ids'][:],
        handle['next_sentence_labels'][:]
    ]
    handle.close()

ind=[(i, j) for idx in range(num_shards) for i, j in zip(cycle([idx]), list(range(ifile_handles[idx][0].shape[0]))) ]
random.shuffle(ind)

# dumps per shard sample indexes
master_sample_idx = 0
for ofile_idx in tqdm(range(num_shards)):
    n_samples_in_this_shard = ifile_handles[ofile_idx][0].shape[0]
    idxs=ind[master_sample_idx:master_sample_idx+n_samples_in_this_shard]
    with open(f'{args.output_hdf5}/shard_list_{ofile_idx:05}.lst','w') as f:
        f.write(json.dumps(idxs))
    #swsok, this line is omitted.
    master_sample_idx = master_sample_idx+n_samples_in_this_shard

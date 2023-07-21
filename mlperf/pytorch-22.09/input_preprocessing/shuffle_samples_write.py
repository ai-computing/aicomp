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
    ifile_handles[ifile_idx] = [
        handle['input_ids'][:],
        handle['input_mask'][:],
        handle['segment_ids'][:],
        handle['masked_lm_positions'][:],
        handle['masked_lm_ids'][:],
        handle['next_sentence_labels'][:]
    ]
    handle.close()

def single_file_write(i):
    with open(f'{args.output_hdf5}/shard_list_{i:05}.lst','r') as f:
        ind=json.load(f)

    max_seq_length = 512
    max_predictions_per_seq = 76
    master_sample_idx=0
    hdf5_compression_method = False
    n_samples_in_this_shard = len(ind)

    h5_writer = h5py.File('{}/part_{:05d}_of_{:05d}.hdf'.format(args.output_hdf5, i, num_shards), 'w')
    input_ids = h5_writer.create_dataset('input_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    segment_ids = h5_writer.create_dataset('segment_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int8')), compression=hdf5_compression_method)
    masked_lm_positions = h5_writer.create_dataset('masked_lm_positions', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    masked_lm_ids = h5_writer.create_dataset('masked_lm_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    next_sentence_labels = h5_writer.create_dataset('next_sentence_labels', data=np.zeros(n_samples_in_this_shard, dtype="int8"), dtype='i1', compression=hdf5_compression_method)

    for o_sample_idx in tqdm(range(n_samples_in_this_shard), total=n_samples_in_this_shard):
        ifile, isample = ind[master_sample_idx]
        input_ids[o_sample_idx] = ifile_handles[ifile][0][isample, :sum(ifile_handles[ifile][1][isample])]
        segment_ids[o_sample_idx] = ifile_handles[ifile][2][isample, :sum(ifile_handles[ifile][1][isample])]
        masked_lm_positions[o_sample_idx] = ifile_handles[ifile][3][isample, :sum(ifile_handles[ifile][3][isample]!=0)]
        masked_lm_ids[o_sample_idx] = ifile_handles[ifile][4][isample, :sum(ifile_handles[ifile][3][isample]!=0)]
        next_sentence_labels[o_sample_idx] = ifile_handles[ifile][5][isample]
        master_sample_idx += 1

    h5_writer.flush()
    h5_writer.close()

from multiprocessing import Pool
import multiprocessing

with Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(single_file_write, range(num_shards))


'''
    with open(f'{args.output_hdf5}/shard_list_{i:05}.lst','r') as f:
        ind=json.load(f)

    max_seq_length = 512
    max_predictions_per_seq = 76
    master_sample_idx=0
    hdf5_compression_method = False
    n_samples_in_this_shard = len(ind)

    h5_writer = h5py.File('{}/part_{:05d}_of_{:05d}.hdf'.format(args.output_hdf5, i, num_shards), 'w')
    input_ids = h5_writer.create_dataset('input_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    segment_ids = h5_writer.create_dataset('segment_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int8')), compression=hdf5_compression_method)
    masked_lm_positions = h5_writer.create_dataset('masked_lm_positions', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    masked_lm_ids = h5_writer.create_dataset('masked_lm_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    next_sentence_labels = h5_writer.create_dataset('next_sentence_labels', data=np.zeros(n_samples_in_this_shard, dtype="int8"), dtype='i1', compression=hdf5_compression_method)

    for o_sample_idx in tqdm(range(n_samples_in_this_shard), total=n_samples_in_this_shard):
        ifile, isample = ind[master_sample_idx]
        input_ids[o_sample_idx] = ifile_handles[ifile][0][isample, :sum(ifile_handles[ifile][1][isample])]
        segment_ids[o_sample_idx] = ifile_handles[ifile][2][isample, :sum(ifile_handles[ifile][1][isample])]
        masked_lm_positions[o_sample_idx] = ifile_handles[ifile][3][isample, :sum(ifile_handles[ifile][3][isample]!=0)]
        masked_lm_ids[o_sample_idx] = ifile_handles[ifile][4][isample, :sum(ifile_handles[ifile][3][isample]!=0)]
        next_sentence_labels[o_sample_idx] = ifile_handles[ifile][5][isample]
        master_sample_idx += 1

    h5_writer.flush()
    h5_writer.close()
'''


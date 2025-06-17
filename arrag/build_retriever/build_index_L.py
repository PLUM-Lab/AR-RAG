# build_faiss_index.py
import os
import glob
import math
import h5py
import faiss
import numpy as np
from tqdm import tqdm
import argparse
import pdb

path_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0], 
                    help="List of GPU IDs to use for FAISS index training and querying.")
args = parser.parse_args()

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
H5_DIR = f'{path_workspace}/data/retriever/img2code_L'
OUTPUT_DIR = f'{path_workspace}/data/retriever/index_L'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRAM_SIZE = 4
EMBED_DIM = GRAM_SIZE * 8
TOKEN_SEQ_LEN = 576
LENGTH_IMG = int(math.sqrt(TOKEN_SEQ_LEN))
MAX_I = TOKEN_SEQ_LEN - LENGTH_IMG - 1

# Adjust these parameters based on your dataset and GPU memory
# NLIST = 80_000
# CHUNK_SIZE = 10_000_000
# IMAGE_CHUNK_SIZE = 10000
# TRAINING_SAMPLE_SIZE = NLIST * 100

NLIST = 800
CHUNK_SIZE = 10000
IMAGE_CHUNK_SIZE = 1000
TRAINING_SAMPLE_SIZE = NLIST * 100

M_SUBQ = 8
BITS_PER_SUBQ = 8
USE_OPQ = True

# --------------------------------------------------------------------------------
# Training and index construction
# --------------------------------------------------------------------------------
def sample_training_data(h5_files, total_sample_size=TRAINING_SAMPLE_SIZE):
    rng = np.random.default_rng(seed=42)
    samples, collected = [], 0

    for h5_file in tqdm(h5_files, desc="Sampling training data"):
        if collected >= total_sample_size:
            break
        with h5py.File(h5_file, 'r') as f:
            vectors = f['vectors'][:]
        N = vectors.shape[0]
        if N == 0: continue

        want = min(total_sample_size - collected, N * MAX_I)
        img_idxs = rng.integers(low=0, high=N, size=want)
        start_idxs = rng.integers(low=0, high=MAX_I, size=want)

        out = np.vstack([
            np.vstack((vectors[im, st:st+GRAM_SIZE-1], vectors[im, st+LENGTH_IMG])).reshape(-1)
            for im, st in zip(img_idxs, start_idxs)
        ])

        samples.append(out)
        collected += want

    return np.vstack(samples)[:total_sample_size]

def train_index(train_data, gpu_resources):
    """
    Train the index on CPU and move it to GPU using provided resources.
    Args:
        train_data: Training data for the index.
        gpu_resources: List of faiss.StandardGpuResources to reuse.
    """
    quantizer = faiss.IndexFlatL2(EMBED_DIM)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, EMBED_DIM, NLIST, M_SUBQ, BITS_PER_SUBQ)

    if USE_OPQ:
        opq_matrix = faiss.OPQMatrix(EMBED_DIM, M_SUBQ)
        index = faiss.IndexPreTransform(opq_matrix, index_ivfpq)
    else:
        index = index_ivfpq

    # Train on CPU
    index.train(train_data)

    # Save initial trained CPU index
    initial_cpu_index_path = os.path.join(OUTPUT_DIR, "initial_trained_cpu.index")
    faiss.write_index(index, initial_cpu_index_path)

    # Move to GPU using existing gpu_resources
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    index_gpu = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)

    return index_gpu, initial_cpu_index_path

def add_data_to_index(index_gpu, h5_files, image_chunk_size=IMAGE_CHUNK_SIZE):
    current_id, all_next_codes, index_part_id = 0, [], 0

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            vectors, indices = f['vectors'][:], f['indices'][:]
        N = vectors.shape[0]
        xb, nb = None, None

        for start_idx in tqdm(range(0, N, image_chunk_size), desc=f"Adding vectors from {h5_file}"):
            end_idx = min(start_idx + image_chunk_size, N)
            batch_vectors = vectors[start_idx:end_idx]
            batch_indices = indices[start_idx:end_idx]

            batch_vectors = batch_vectors.reshape(-1, 24, 24, 8)
            batch_indices = batch_indices.reshape(-1, 24, 24)

            horizontal_grams = np.array([
                np.lib.stride_tricks.sliding_window_view(
                    batch_vectors[:, r, :, :], window_shape=(3,), axis=1
                ).transpose(0, 1, 3, 2)
                for r in range(23)
            ])

            vertical_vectors = batch_vectors[:, 1:24, 0:22, :]

            gram_embeds = np.concatenate(
                [horizontal_grams.transpose(1, 0, 2, 3, 4), vertical_vectors[:, :, :, None, :]], axis=3
            )

            xb_batch = gram_embeds.reshape(-1, 32)
            nb_batch = batch_indices[:, 1:24, 1:23].reshape(-1)

            if xb is None:
                xb, nb = xb_batch, nb_batch
            else:
                xb = np.vstack((xb, xb_batch))
                nb = np.concatenate((nb, nb_batch))

            if xb.shape[0] >= CHUNK_SIZE:
                ids = np.arange(current_id, current_id + len(xb))
                index_gpu.add_with_ids(np.array(xb, dtype=np.float32), ids)
                all_next_codes.extend(nb)
                current_id += len(xb)
                xb, nb = None, None

        print(f"Finished processing {h5_file}, Current ID: {current_id}")

    # Save any remaining data
    if all_next_codes:
        final_cpu_index = faiss.index_gpu_to_cpu(index_gpu)
        faiss.write_index(final_cpu_index, os.path.join(
            OUTPUT_DIR, f"faiss_ivf_pq_opq.index"))
        np.save(os.path.join(
            OUTPUT_DIR, f"next_codes.npy"), np.array(all_next_codes))
        print(f"Saved final index {index_part_id}.")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    h5_files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))

    # Create gpu_resources once at the beginning
    gpu_resources = [faiss.StandardGpuResources() for _ in args.gpu_ids]

    train_data = sample_training_data(h5_files)
    index_gpu, initial_cpu_index_path = train_index(train_data, gpu_resources)
    # initial_cpu_index_path = (/local/initial_trained_cpu.index)
    # initial_cpu_index = faiss.read_index(initial_cpu_index_path)
    # co = faiss.GpuMultipleClonerOptions()
    # co.shard = True
    # index_gpu = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, initial_cpu_index, co)
    add_data_to_index(index_gpu, h5_files)

if __name__ == '__main__':
    main()

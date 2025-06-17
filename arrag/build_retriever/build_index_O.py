import os
import glob
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
H5_DIR = f'{path_workspace}/data/retriever/img2code_O'
OUTPUT_DIR = f'{path_workspace}/data/retriever/index_O'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 8 * 13  # 8 surrounding patches, each patch 13-dimensional
TOKEN_SEQ_LEN = 256
LENGTH_IMG = int(np.sqrt(TOKEN_SEQ_LEN))

NLIST = 14830
CHUNK_SIZE = 1000_000
IMAGE_CHUNK_SIZE = 256
TRAINING_SAMPLE_SIZE = NLIST * 100

M_SUBQ = 8  # Changed from 16 to 8 to ensure EMBED_DIM (104) is divisible by M_SUBQ
BITS_PER_SUBQ = 8
USE_OPQ = True

# --------------------------------------------------------------------------------
# Helper function to extract 0-shaped patches (vectorized)
# --------------------------------------------------------------------------------
def extract_o_shape_patches(batch_vectors):
    B, H, W, D = batch_vectors.shape  # B, 16, 16, 13
    # Pad the grid with zeros on all sides
    padded = np.pad(batch_vectors, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
    
    # Extract all surrounding patches using slicing
    ul = padded[:, 0:H, 0:W, :]       # upper left
    u = padded[:, 0:H, 1:W+1, :]     # upper
    ur = padded[:, 0:H, 2:W+2, :]    # upper right
    l = padded[:, 1:H+1, 0:W, :]     # left
    r = padded[:, 1:H+1, 2:W+2, :]   # right
    ll = padded[:, 2:H+2, 0:W, :]    # lower left
    lo = padded[:, 2:H+2, 1:W+1, :]  # lower
    lr = padded[:, 2:H+2, 2:W+2, :]  # lower right
    
    # Concatenate along the last dimension to form the O-shape patch vectors
    patches = np.concatenate([ul, u, ur, l, r, ll, lo, lr], axis=-1)
    
    # Reshape to (B, H, W, 8 * D) and then to (B * H * W, 8 * D)
    patches = patches.reshape(B, H, W, 8 * D)
    return patches.reshape(-1, 8 * D)

# --------------------------------------------------------------------------------
# Sample training data (updated to extract 0-shaped patches)
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

        for img_idx in rng.choice(N, size=min(N, (total_sample_size - collected) // 256), replace=False):
            batch_vectors = vectors[img_idx].reshape(1, LENGTH_IMG, LENGTH_IMG, 13)
            patch_vectors = extract_o_shape_patches(batch_vectors)
            samples.append(patch_vectors)
            collected += patch_vectors.shape[0]
            if collected >= total_sample_size:
                break

    all_samples = np.vstack(samples)
    return all_samples[:total_sample_size]

# --------------------------------------------------------------------------------
# Training function remains unchanged
# --------------------------------------------------------------------------------
def train_index(train_data, gpu_resources):
    quantizer = faiss.IndexFlatL2(EMBED_DIM)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, EMBED_DIM, NLIST, M_SUBQ, BITS_PER_SUBQ)

    if USE_OPQ:
        opq_matrix = faiss.OPQMatrix(EMBED_DIM, M_SUBQ)
        index = faiss.IndexPreTransform(opq_matrix, index_ivfpq)
    else:
        index = index_ivfpq

    index.train(train_data)
    initial_cpu_index_path = os.path.join(OUTPUT_DIR, "initial_trained_cpu.index")
    faiss.write_index(index, initial_cpu_index_path)

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    index_gpu = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)

    return index_gpu, initial_cpu_index_path

# --------------------------------------------------------------------------------
# Add data to index (updated for O-shape)
# --------------------------------------------------------------------------------
def add_data_to_index(index_gpu, initial_cpu_index_path, h5_files, gpu_resources):
    current_id, all_next_codes, index_part_id = 0, [], 0

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            vectors = f['vectors'][:]
            indices = f['indices'][:]
        N = vectors.shape[0]

        xb, nb = [], []
        for start_idx in tqdm(range(0, N, IMAGE_CHUNK_SIZE), desc=f"Adding vectors from {h5_file}"):
            end_idx = min(start_idx + IMAGE_CHUNK_SIZE, N)
            batch_vectors = vectors[start_idx:end_idx].reshape(-1, LENGTH_IMG, LENGTH_IMG, 13)
            batch_indices = indices[start_idx:end_idx].reshape(-1, LENGTH_IMG, LENGTH_IMG)

            xb_batch = extract_o_shape_patches(batch_vectors)
            nb_batch = batch_indices.reshape(-1)

            xb.append(xb_batch)
            nb.append(nb_batch)

            if np.concatenate(xb, axis=0).shape[0] >= CHUNK_SIZE:
                xb_concat = np.concatenate(xb, axis=0)
                nb_concat = np.concatenate(nb, axis=0)
                ids = np.arange(current_id, current_id + len(xb_concat))
                index_gpu.add_with_ids(xb_concat.astype(np.float32), ids)
                all_next_codes.extend(nb_concat)
                current_id += len(xb_concat)
                xb, nb = [], []

        print(f"Finished processing {h5_file}, Current ID: {current_id}")

    if xb:
        xb_concat = np.concatenate(xb, axis=0)
        nb_concat = np.concatenate(nb, axis=0)
        ids = np.arange(current_id, current_id + len(xb_concat))
        index_gpu.add_with_ids(xb_concat.astype(np.float32), ids)
        all_next_codes.extend(nb_concat)

    final_cpu_index = faiss.index_gpu_to_cpu(index_gpu)
    faiss.write_index(final_cpu_index, os.path.join(
        OUTPUT_DIR, f"faiss_ivf_pq_opq.index"))
    np.save(os.path.join(
        OUTPUT_DIR, f"next_codes.npy"), np.array(all_next_codes))

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    h5_files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
    gpu_resources = [faiss.StandardGpuResources() for _ in args.gpu_ids]

    train_data = sample_training_data(h5_files)
    index_gpu, initial_cpu_index_path = train_index(train_data, gpu_resources)
    add_data_to_index(index_gpu, initial_cpu_index_path, h5_files, gpu_resources)

if __name__ == '__main__':
    main()
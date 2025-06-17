#!/usr/bin/env python

import os
import faiss
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaissRetrieverAllIndexes:
    """
    Loads all partial_{i}.index into memory as separate CPU-based indexes.
    Also loads or builds a single next_codes array so next_codes[global_id] -> code.

    At query time, we:
      - (Multi-threaded) loop over each in-memory index
      - search individually
      - collect results (D, I) across all indexes
      - partial-sort for top_k
      - map IDs to codes
    """

    def __init__(
        self,
        base_path: str,
        dim: int,
        top_k: int = 10,
        use_gpu: bool = False,
        dim_vocab: int = 16384,
        nprobe: int = 32,       # you can tweak or pass an argument
        device_id: int = None
    ):
        """
        Args:
            base_path: Path containing partial_{i}.index + next_codes_{i}.npy (or merged).
            shards_num: Number of partial indexes.
            dim: Dimension of embeddings (e.g. 32).
            top_k: How many neighbors to retrieve per query.
            use_gpu: (Optional) If you want each index to be moved to GPU at load time.
            dim_vocab: For the optional get_distribution() usage.
            nprobe: Number of clusters to probe for IVF indexes.
            thread_search: If True, we use a ThreadPoolExecutor to parallelize searches.
            num_threads: Number of parallel threads in search.
        """
        self.base_path = base_path
        self.dim = dim
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.dim_vocab = dim_vocab
        self.nprobe = nprobe
        self.device_id = device_id

        logger.info("Loading all partial indexes into memory (no merging)...")
        self.indexes, self.next_codes = self._load_all_indexes()

        logger.info("Initialization complete. All partial indexes + next_codes in memory.")
        # pdb.set_trace()

    def _load_all_indexes(self):
        """
        Reads partial_{i}.index from disk, storing them in a list [idx_0, idx_1, ...].
        Optionally moves them to GPU if self.use_gpu = True (be sure you have enough GPU memory).
        Also sets nprobe for IVF-based indexes.
        """
        # Load the CPU index
        cpu_index = faiss.read_index(os.path.join(self.base_path, f"faiss_ivf_pq_opq.index"))

        # Move index to GPUs (three A30 GPUs)
        gpu_resources = [faiss.StandardGpuResources() for _ in self.device_id]
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # distribute across GPUs
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, cpu_index, co)
        # set nprobe
        gpu_index.nprobe = self.nprobe
        codes = np.load(os.path.join(self.base_path, "next_codes.npy"))
        logger.info(f"Total codes loaded: {len(codes)/1e6:.2f} million")
        return gpu_index, codes

    def search(self, query: np.ndarray, construct_train=False, topk=0, unique=False, reshape=True):
        """
        Multi-threaded (or single-thread) search over all in-memory indexes:
          1) For each index, do index.search(query, top_k).
          2) Vectorize mapping of IDs -> next_codes (instead of for-loop).
          3) Merge partial results => shape (n_queries, #shards * top_k).
          4) Partial sort => global top_k.
          5) Build distribution (optional).
        """
        if topk != 0:
            self.top_k = topk

        # Ensure shape = (n_queries, dim)
        if reshape:
            if query.shape[0] != 1:
                query = query.reshape(1, -1)

        n_queries = query.shape[0]

        D, I = self.indexes.search(query, k=self.top_k)
        I = np.take(self.next_codes, I)
        final_distances_t = torch.tensor(D)
        final_codes_t = torch.tensor(I)

        # 5) If you want a distribution for the first query only:
        if construct_train:
            return final_distances_t, final_codes_t

        distribution = None
        if n_queries > 0:
            # create distribution from row 0’s codes & distances
            if unique:
                codes_row_t = final_codes_t[0]
                distances_row_t = final_distances_t[0]
                # Sort indices and track original positions
                original_indices = torch.arange(codes_row_t.size(0), device=codes_row_t.device)
                sorted_values, sorted_indices = torch.sort(codes_row_t)
                sorted_original_indices = original_indices[sorted_indices]

                # Find first occurrence indices in original array
                unique_values_sorted, inverse_sorted = torch.unique(sorted_values, return_inverse=True)
                group_min_indices = torch.full_like(unique_values_sorted, codes_row_t.size(0), dtype=torch.long) 
                # Use scatter_reduce_ with amin 
                group_min_indices.scatter_reduce_( 0, inverse_sorted, sorted_original_indices, reduce='amin', include_self=False )

                # Create mask for first occurrences
                mask = torch.zeros_like(codes_row_t, dtype=torch.bool)
                mask[group_min_indices] = True

                # Extract unique values with original order
                final_codes_t = codes_row_t[mask]
                final_distances_t = distances_row_t[mask]
                return final_distances_t, final_codes_t
            else:
                distribution = self._get_distribution_gpu(final_distances_t[0], final_codes_t[0])


        # Return the final Tensors or convert them to CPU if you prefer
        # E.g. as CPU numpy:
        final_distances = final_distances_t.cpu().numpy()  # shape (n_queries, top_k)
        final_codes = final_codes_t.cpu().numpy()          # shape (n_queries, top_k)

        return final_distances, final_codes_t, final_codes, distribution

    def _get_distribution_gpu(self, distances_row_t: torch.Tensor, codes_row_t: torch.Tensor, unique=False) -> torch.Tensor:
        """
        distances_row_t: Tensor of shape (top_k,) on GPU
        codes_row_t:     Tensor of shape (top_k,) on GPU
        Return:          Tensor of shape (dim_vocab,) on GPU
        """
        # 1) Convert distances -> weights
        #    shape = (top_k,)
        exp_weights = torch.exp(-distances_row_t)

        if unique:
            # Sort indices and track original positions
            original_indices = torch.arange(codes_row_t.size(0), device=codes_row_t.device)
            sorted_values, sorted_indices = torch.sort(codes_row_t)
            sorted_original_indices = original_indices[sorted_indices]

            # Find first occurrence indices in original array
            unique_values_sorted, inverse_sorted = torch.unique(sorted_values, return_inverse=True)
            group_min_indices = torch.full_like(unique_values_sorted, codes_row_t.size(0), dtype=torch.long) 
            # Use scatter_reduce_ with amin 
            group_min_indices.scatter_reduce_( 0, inverse_sorted, sorted_original_indices, reduce='amin', include_self=False )

            # Create mask for first occurrences
            mask = torch.zeros_like(codes_row_t, dtype=torch.bool)
            mask[group_min_indices] = True

            # Extract unique values with original order
            codes_row_t = codes_row_t[mask]
            exp_weights = exp_weights[mask]
            # get std
            std_weights = torch.std(exp_weights)
            base_weight = exp_weights.max() / (1 + std_weights)
            final_weight = base_weight * torch.sigmoid(exp_weights.mean() - 0.5)

        # 2) Create an empty distribution = min_weight for each code in [0..dim_vocab-1]
        distribution = torch.zeros(self.dim_vocab, device=distances_row_t.device)

        # 3) Accumulate
        distribution.index_add_(0, codes_row_t, exp_weights)
        # pdb.set_trace()
        # 4) Normalize
        s = distribution.sum()
        if s > 0:
            distribution /= s

        if unique:
            return distribution, final_weight
        return distribution
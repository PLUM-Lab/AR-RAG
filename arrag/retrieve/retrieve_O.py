#!/usr/bin/env python

import os
import faiss
import numpy as np
import logging
import time
import torch
import pdb
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaissRetrieverAllIndexes:
    """
    Loads all partial_{i}.index into memory as separate CPU-based indexes.
    Also loads or builds a single next_codes array so next_codes[global_id] -> code.
    """

    def __init__(
        self,
        base_path: str,
        dim: int,
        top_k: int = 10,
        use_gpu: bool = False,
        dim_vocab: int = 16384,
        nprobe: int = 4096,
        device_id: int = None
    ):
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

    def _load_all_indexes(self):
        cpu_index = faiss.read_index(os.path.join(self.base_path, f"faiss_ivf_pq_opq.index"))
        gpu_resources = [faiss.StandardGpuResources() for _ in self.device_id]
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, cpu_index, co)
        gpu_index.nprobe = self.nprobe
        codes = np.load(os.path.join(self.base_path, "next_codes.npy"))
        logger.info(f"Total codes loaded: {len(codes)/1e6:.2f} million")
        return gpu_index, codes

    def search(self, query: np.ndarray, construct_train=False, topk=0):
        """
        Search over all in-memory indexes, supporting multiple queries.
        """
        if topk != 0:
            self.top_k = topk

        # Ensure query is 2D: [n_queries, dim]
        if query.ndim == 1:
            query = query.reshape(1, -1)

        n_queries = query.shape[0]

        D, I = self.indexes.search(query, k=self.top_k)
        I = np.take(self.next_codes, I)
        final_distances_t = torch.tensor(D)
        final_codes_t = torch.tensor(I)

        return final_distances_t, final_codes_t

    def _get_distribution_gpu(self, distances_row_t: torch.Tensor, codes_row_t: torch.Tensor, unique=False) -> torch.Tensor:
        exp_weights = torch.exp(-distances_row_t)
        if unique:
            original_indices = torch.arange(codes_row_t.size(0), device=codes_row_t.device)
            sorted_values, sorted_indices = torch.sort(codes_row_t)
            sorted_original_indices = original_indices[sorted_indices]
            unique_values_sorted, inverse_sorted = torch.unique(sorted_values, return_inverse=True)
            group_min_indices = torch.full_like(unique_values_sorted, codes_row_t.size(0), dtype=torch.long)
            group_min_indices.scatter_reduce_(0, inverse_sorted, sorted_original_indices, reduce='amin', include_self=False)
            mask = torch.zeros_like(codes_row_t, dtype=torch.bool)
            mask[group_min_indices] = True
            codes_row_t = codes_row_t[mask]
            exp_weights = exp_weights[mask]
            std_weights = torch.std(exp_weights)
            base_weight = exp_weights.max() / (1 + std_weights)
            final_weight = base_weight * torch.sigmoid(exp_weights.mean() - 0.5)

        distribution = torch.zeros(self.dim_vocab, device=distances_row_t.device)
        distribution.index_add_(0, codes_row_t, exp_weights)
        s = distribution.sum()
        if s > 0:
            distribution /= s

        if unique:
            return distribution, final_weight
        return distribution
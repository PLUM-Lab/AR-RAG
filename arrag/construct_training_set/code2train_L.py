import os
import sys

import torch
from transformers import AutoModelForCausalLM

path_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{path_workspace}/arrag/Janus')
from janus.models import MultiModalityCausalLM
sys.path.append(f'{path_workspace}/arrag/retrieve')
from retrieve_L import FaissRetrieverAllIndexes
from datasets import load_dataset
import argparse

def retrieve(example):
    """
    Vectorized retrieve function that queries all valid image-token positions
    in a single FAISS call.
    """
    device = torch.device('cuda')
    image_tokens = torch.tensor(example['code.npy'], device=device)  # shape (576,)

    # 1) Identify all the valid positions i
    #    i.e. the ones that pass your if-statement:
    #        if i > image_dim and (i-1) % image_dim < image_dim - retri_num - 1
    valid_indices = []
    for i in range(image_token_num_per_image):
        if (i > image_dim) and ((i - 1) % image_dim < (image_dim - retri_num - 1)):
            valid_indices.append(i)

    # 2) Build the batch of 4 tokens per valid position
    #    (From your snippet: new_tokens_L was the concatenation of
    #     image_tokens[i-image_dim-1:i-image_dim+2] and image_tokens[i-1].unsqueeze(0))
    #    That’s effectively 4 tokens: [-image_dim-1, -image_dim, -image_dim+1, -1].
    #    Adjust if your logic changes.
    all_token_groups = []
    for i in valid_indices:
        new_tokens_L = torch.cat([
            image_tokens[i - image_dim - 1 : i - image_dim + 2],  # 3 tokens
            image_tokens[i - 1].unsqueeze(0)                      # 4th token
        ], dim=0)
        all_token_groups.append(new_tokens_L)

    if len(all_token_groups) == 0:
        # If somehow no valid positions, just set empty retrieval
        example['retrieved_codes.npy'] = [None]*image_token_num_per_image
        example['retrieved_scores.npy'] = [None]*image_token_num_per_image
        return example

    # Stack to shape [num_valid, 4]
    all_token_groups_t = torch.stack(all_token_groups, dim=0)  # shape (num_valid, 4)

    # 3) Run VQ codebook in batch => shape (num_valid, 4, embed_dim)
    with torch.no_grad():
        vq_embeddings = vq_model.quantize.get_codebook_entry(all_token_groups_t)
        # For example, if each token is an 8D embedding, vq_embeddings is (num_valid, 4, 8).

    # 4) Flatten => shape (num_valid, 4*embed_dim)
    #    The FAISS index dimension is 32 in your example; presumably 4 * 8 = 32
    vq_embeddings = vq_embeddings.view(vq_embeddings.size(0), -1)  # (num_valid, 32)

    # Move to CPU Numpy (FAISS typically wants CPU float32 if you’re using the retriever’s .search)
    query = vq_embeddings.float().cpu().numpy()  # shape (num_valid, 32)

    # 5) Single search call for the entire batch
    #    Here we use construct_train=False so that we get results for *all* queries (not only row 0).
    #    The return signature is:
    #      final_distances, final_codes, _, distribution
    #    We only need final_distances & final_codes for each query.
    final_distances, final_codes = retriever.search(
        query, construct_train=True, topk=10, reshape=False
    )
    # final_distances => shape (num_valid, top_k)
    # final_codes     => shape (num_valid, top_k)
    # We can convert them to CPU again or keep them in numpy. They should already be CPU arrays.

    # 6) Convert distances -> "scores"
    #    If you want the same “torch.exp(-distances)” logic, just do that in numpy:
    scores_np = torch.exp(-final_distances).cpu().numpy()  # shape (num_valid, top_k)

    # 7) Insert these results back into a 576-length list, placing None for invalid i
    retrieved_codes = [None] * image_token_num_per_image
    retrieved_scores = [None] * image_token_num_per_image

    for idx, i in enumerate(valid_indices):
        retrieved_codes[i] = final_codes[idx]   # shape (top_k,)
        retrieved_scores[i] = scores_np[idx]    # shape (top_k,)

    example['retrieved_codes.npy'] = retrieved_codes
    example['retrieved_scores.npy'] = retrieved_scores
    return example

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--device_id', type=list, default=[0])

    args = parser.parse_args()

    # load VAE model
    model_path = "deepseek-ai/Janus-Pro-1B"
    vq_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).gen_vision_model
    device = torch.device(f"cuda")
    vq_model = vq_model.to(torch.bfloat16).to(device).eval()

    # load retriever
    retriever = FaissRetrieverAllIndexes(
        base_path=os.path.join(path_workspace, args.retriever_path),
        dim=32,
        top_k=10,
        use_gpu=True,
        dim_vocab=16384,
        nprobe=2048,
        device_id=args.device_id,
    )

    # load training dataset
    train_dataset = load_dataset(
        'webdataset', 
        data_files=f'{args.data_path}/*.tar', 
        split="train", 
        streaming=False,
        num_proc=64,
    )
    # random sample 1000 images for testing
    train_dataset = train_dataset.shuffle()
    # train_dataset = train_dataset.select(range(250000)) # for testing script

    image_token_num_per_image: int = 576
    img_size: int = 384
    patch_size: int = 16
    image_dim = img_size // patch_size
    retri_num: int = 1

    train_dataset = train_dataset.map(
        retrieve,
    )

    save_path = os.path.join(path_workspace, args.save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    train_dataset.save_to_disk(
        save_path,
        max_shard_size="4GB"
    )

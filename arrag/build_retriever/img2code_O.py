import os
import sys
import torch
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import torchvision
from datasets import load_dataset
from torch.utils.data import DataLoader
import shutil

path_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{path_workspace}/arrag/Janus')
from models import MAGVITv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_source', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='data/retriever/img2code_O', help='Path to save the HDF5 file')
    parser.add_argument('--shards_idx', type=int, default=0, help='Shard index for parallel processing')
    parser.add_argument('--shards_num', type=int, default=1, help='Total number of shards to split the source dataset')
    parser.add_argument('--num_processors', type=int, default=64)  # adjust based on your system
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    assert args.shards_num > args.shards_idx >= 0, "shards_num must be greater than shards_idx and both must be non-negative."

    # Where we save the HDF5 file
    if os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    save_path = f'{path_workspace}/{args.save_path}/shard_{args.shards_idx}.h5'
    if os.path.exists(save_path):
        print(f"Shard {args.shards_idx} already processed, skipping.")
        return

    # Create a custom cache directory for this process
    # so we can delete it safely afterward.
    cache_dir = os.path.join(path_workspace, f'{args.save_path}/tmp_hf_cache', f'shard_{args.shards_idx}')
    os.makedirs(cache_dir, exist_ok=True)
    
    # -------------------------
    # 1) Prepare the device
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # -------------------------
    # 2) Define image transform
    # -------------------------
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # -------------------------
    # 3) Set up the VQ model
    # -------------------------
    vq_model = MAGVITv2.from_pretrained('showlab/magvitv2').to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    
    # -------------------------
    # 4) Build the dataset from all Parquet files in input_dir
    # ----------------------------------------------------------------
    ds = load_dataset(
        args.img_source,
        split="train",
        cache_dir=cache_dir,
    )
    ds = ds.shuffle(seed=42)
    # reserve 1/10 data for testing
    ds = ds.select(range(len(ds) // 10, len(ds)))
    ds = ds.select(range(args.shards_idx, len(ds), args.shards_num))
    total_num = len(ds)
    print(f"Dataset loaded with {total_num} samples.")
    
    # ------------------------------------------------------------------
    # 5) Define a collate function for on-the-fly transforms
    # ------------------------------------------------------------------
    def collate_fn(examples):
        """
        Each `example` is a dict. The 'jpg' entry may be a PIL image 
        (if HF automatically decodes) or raw bytes. If it's bytes, 
        you'd need to do: 
            img = Image.open(io.BytesIO(ex['jpg']))
        We'll assume it's a PIL image here. 
        """
        imgs = []
        for ex in examples:
            # Ensure RGB
            pil_img = ex['image'].convert('RGB')
            # Apply your transform
            tensor_img = transform(pil_img)
            imgs.append(tensor_img)
        # Stack into a single tensor [B, C, H, W]
        return {'image_tensor': torch.stack(imgs, dim=0)}

    # ------------------------------------------------------------------
    # 6) Wrap in DataLoader for parallel decode + transform
    # ------------------------------------------------------------------
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_processors,   # parallel reading + transform
        pin_memory=(device == 'cuda'),
        collate_fn=collate_fn
    )
    print("Created DataLoader with on-the-fly transforms in collate_fn.")

    all_codes = []
    all_embeddings = []
    total_steps = (total_num // args.batch_size + 1) if total_num else None

    # ----------------------------------
    # 7) Encode images in batches
    # ----------------------------------
    with torch.no_grad():
        for i_data, batch in enumerate(tqdm(loader, total=total_steps, desc=f"Shard {args.shards_idx}")):
            images = batch['image_tensor'].to(device, non_blocking=True)
            
            # Encode
            quantized_states, codebook_indices = vq_model.encode(images)
            codebook_indices = codebook_indices.cpu().numpy()
            quantized_states = torch.einsum("b c h w -> b h w c", quantized_states).contiguous() # [batch, h, w, c]
            quantized_states = quantized_states.reshape(images.size(0), -1, 13).cpu().numpy() # [batch, h*w, c]
            
            all_codes.append(codebook_indices)
            all_embeddings.append(quantized_states)

    # ----------------------------------
    # 8) Concatenate all results
    # ----------------------------------
    total_codes = np.concatenate(all_codes, axis=0)
    total_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # ----------------------------------
    # 9) Save results to HDF5
    # ----------------------------------
    with h5py.File(save_path, 'w') as f:
        f.create_dataset(
            'indices',
            data=total_codes,
            dtype='int32',
            compression='gzip',
            compression_opts=9
        )
        f.create_dataset(
            'vectors',
            data=total_embeddings,
            dtype='float32',
            compression='gzip',
            compression_opts=9
        )
        
        # Metadata
        f.attrs['num_pairs'] = len(total_embeddings)
        if len(total_embeddings.shape) == 3:
            f.attrs['sequence_length'] = total_embeddings.shape[1]
            f.attrs['vector_dim'] = total_embeddings.shape[2]
    
    print(f"Finished shard {args.shards_idx}, saved to {save_path}.")

    # -------------------------
    # 10) Clean up the cache
    # -------------------------
    print(f"Removing cache for shard {args.shards_idx} at {cache_dir} ...")
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("Done removing cache.")


if __name__ == "__main__":
    main()

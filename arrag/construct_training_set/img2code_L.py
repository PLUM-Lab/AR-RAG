import os
import sys
import glob
import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from datasets import load_dataset
import webdataset as wds
from tqdm import tqdm

path_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{path_workspace}/arrag/Janus')
from janus.models.vq_model import VQ_models
import pdb

# Apply your transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(384),
    torchvision.transforms.CenterCrop(384),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def collate_fn(samples):
    """
    Collate function for the DataLoader.
      - 'samples' is a list of dicts from the HF dataset,
        each containing {'image_bytes': ..., 'caption': ...}.
      - We decode and transform images in batch, return a dict
        with 'images' (tensor) and 'captions' (list of strings).
    """
    pil_images = []
    captions = []
    for s in samples:
        pil_images.append(s['image'].convert("RGB"))
        # pil_images.append(Image.open(io.BytesIO(s["image"])).convert("RGB")) # for midjourney data
        captions.append(s["caption"])

    # Transform each PIL image
    tensors = [transform(img) for img in pil_images]
    # Stack into a single tensor [B, C, H, W]
    images_tensor = torch.stack(tensors, dim=0)

    return {
        "images": images_tensor,
        "captions": captions
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_source", type=str, required=True,
                        help="Source image dataset from Hugging Face.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where tar shards will be saved.")
    parser.add_argument("--maxsize", type=int, default=2.5e10,
                        help="Max shard size in bytes (approx) before splitting.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=48,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--txt_field", type=str)
    parser.add_argument('--chunk_idx', type=int, default=0, help='Chunk index for parallel processing')
    parser.add_argument('--chunk_num', type=int, default=1, help='Total number of Chunk to split the source dataset')
    parser.add_argument('--vq_model', type=str, default='arrag/Janus/janus/vq_ds16_t2i.pt')
    args = parser.parse_args()

    os.makedirs(os.path.join(path_workspace, args.output_dir), exist_ok=True)

    # ----------------------------------------------------------------
    # 1) Prepare device
    # ----------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------------
    # 2) Set up the VQ model
    # ----------------------------------------------------------------
    vq_model = VQ_models['VQ-16'](codebook_size=16384, codebook_embed_dim=8)
    checkpoint = torch.load(
        args.vq_model,
        map_location="cpu"
    )
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    vq_model.to(device)
    vq_model.eval()
    print("VQ model loaded.")

    # ----------------------------------------------------------------
    # 3) Build the dataset from all Parquet files in img_source
    # ----------------------------------------------------------------
    ds = load_dataset(
        args.img_source,
        split="train",
    )
    ds = ds.shuffle(seed=42)
    # use the 1/10 data for testing
    ds = ds.select(range(len(ds) // 10))

    print(f"Dataset loaded with {len(ds)} samples.")
    
    if args.txt_field == 'conversations':
        def transform_example(ex):
            return {
                "image": ex["image"],
                "caption": ex["conversations"][1]["value"]
            }
    else:
        def transform_example(ex):
            return {
                "image": ex["image"],
                "caption": ex["llava"]
            }

    # Apply the transform to each row (this is HF .map, not the same as the PyTorch transform!)
    # remove_columns just drops unneeded columns to keep dataset smaller in memory.
    old_columns = ds.column_names
    ds = ds.map(transform_example, remove_columns=old_columns, num_proc=args.num_workers)

    # We'll rely on a PyTorch DataLoader for batch + shuffle.
    ds.set_format(type=None)  # We'll do custom collation
    dataset_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # ----------------------------------------------------------------
    # 4) Create the WebDataset ShardWriter
    # ----------------------------------------------------------------
    # This will automatically roll over to a new tar shard once
    # maxsize (in bytes) is exceeded.
    shard_pattern = args.output_dir + f"/chunk_{args.chunk_idx}_" + "shard-%06d.tar"
    writer = wds.ShardWriter(shard_pattern, maxsize=args.maxsize)

    # ----------------------------------------------------------------
    # 5) Encode data in batches and write out
    # ----------------------------------------------------------------
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataset_loader, desc="Processing batches"):
            images = batch["images"].to(device, non_blocking=True)
            captions = batch["captions"]

            # VQ encode
            _, _, info = vq_model.encode(images)
            # info[2] is the code (tensor, shape [B, h*w])
            codes = info[2].reshape(images.size(0), -1).cpu().numpy()
            # Write each sample to the tar
            for i in range(images.size(0)):
                sample_dict = {
                    "__key__": f"{global_idx:07d}",  # unique sample key
                    # Storing as "code": (numpy array). 
                    # You can also do "code.npy" if you want standard .npy extension
                    "code.npy": codes[i],  
                    "caption.txt": captions[i]
                }
                writer.write(sample_dict)
                global_idx += 1

    writer.close()
    print(f"Finished processing {global_idx} samples. Tar shards saved in {args.output_dir}.")


if __name__ == "__main__":
    main()

# AR-RAG: Autoregressive Retrieval Augmentation for Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.arxiv.org/abs/2506.06962)
[![Hugging Face Models](https://img.shields.io/badge/🤗_Hugging_Face-Models-orange.svg)](https://huggingface.co/collections/jingyq1/ar-rag-683ba502136fa9ca71e7b2b4)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/PLUM-Lab/AR-RAG)](https://github.com/PLUM-Lab/AR-RAG)

This repository contains the official implementation of [AR-RAG: Autoregressive Retrieval Augmentation for Image Generation](https://www.arxiv.org/abs/2506.06962).

![AR-RAG Showcase](https://github.com/jingyq1/AR-RAG/blob/main/images/idea2.png)

## Contents
- [Overview](#overview)
- [Model Zoo](#model-zoo)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview

AR-RAG introduces a novel retrieval augmentation paradigm that enhances modern photorealistic image generation by augmenting image predictions with k-nearest neighbor (k-NN) retrievals at the patch level. Unlike existing approaches that rely on full-image retrieval conditioned on textual captions, AR-RAG retrieves locally similar patches based on their surrounding visual context, enabling caption-free retrieval while enforcing spatial coherence and semantic consistency for higher-quality image generation.

We propose two parallel frameworks:

1. **Distribution-Augmentation in Decoding (DAiD)**: A training-free decoding strategy that directly merges the distribution of model-predicted patches with the distribution of retrieved patches.

2. **Feature-Augmentation in Decoding (FAiD)**: A parameter-efficient fine-tuning method that smoothly integrates retrieved patches into the generation process via convolution operations.

### Performance Highlights

Our methods significantly improve image generation quality across multiple benchmarks:

#### GenEval Benchmark

| Method | Single Obj. | Two Obj. | Counting | Colors | Position | Color Attri. | Overall ↑ |
|--------|-------------|----------|----------|--------|----------|--------------|-----------|
| Janus-Pro | 0.98 | 0.77 | 0.52 | 0.84 | 0.61 | 0.55 | 0.71 |
| DAiD (ours) | 0.98 | 0.82 | 0.54 | 0.87 | 0.63 | 0.49 | 0.72 |
| FAiD (ours) | 1.00 | 0.92 | 0.41 | 0.87 | 0.71 | 0.60 | 0.75 |

#### DPG-Bench

| Method | Global | Entity | Attribute | Relation | Other | Overall ↑ |
|--------|-------------|----------|----------|--------|----------|--------------|
| Janus-Pro | 81.76 | 84.53 | 84.34 | 92.22 | 75.20 | 77.26 |
| DAiD (ours) | 83.58 | 84.46 | 84.76 | 91.49 | 76.40 | 77.88 |
| FAiD (ours) | 82.67 | 85.80 | 85.38 | 92.3 | 76.80 | 79.36 |

#### MSCOCO and Midjourney Benchmarks (FID ↓)

| Model | MSCOCO FID | Midjourney FID |
|-------|------------|----------------|
| Janus-Pro | 19.59 | 12.81 |
| DAiD (ours) | 18.02 | 11.93 |
| FAiD (ours) | 17.60 | 9.31 |

## Model Zoo

| Model | Description | Size | HF Link |
|-------|-------------|------|---------|
| AR-RAG-FAiD | Fine-tuned model with Smoothly Feature Blending | 1.2B | [🤗 Model](https://huggingface.co/jingyq1/arrag_faid) |

## Patch-level Retrieval Database

| Data Source | Image Num | Suggest GPU Memory | HF Link |
|-------|-------------|------|---------|
| JourneyDB | 1M | 12 GB | [ZIP](http://nlplab1.cs.vt.edu/~jingyuan/AR-RAG/retrieval_db.zip) |
| CC12M | 12M | 96 GB | [ZIP](http://nlplab1.cs.vt.edu/~jingyuan/AR-RAG/retrieval_db.zip) |
| DataCamp | 70M | - | 🤗 Coming soon |

## Installation

```bash
git clone https://github.com/PLUM-Lab/AR-RAG.git
cd AR-RAG

# Create and activate conda environment
conda env create -f arrag.yml
```

## Patch-level Retrieval Database & Retriever Construction

### Download the checkpoint of VQ-VAE model from [LlamaGen](https://github.com/FoundationVision/LlamaGen)

```bash
wget -P arrag/Janus/janus https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt
```

### Construct Retreiver from Image Data

```bash
bash arrag/build_retriever/build_retriever.sh
```

The output faiss index will be: `data/retriever/index_L`

### Download Pre-built Retrieval Database
```bash
# Download pre-built retrieval database
wget http://nlplab1.cs.vt.edu/~jingyuan/AR-RAG/retrieval_db.zip
```

## Training

### FAiD Model Training

```bash
bash ./arrag/train/train_FAiD.sh
```

The default output checkpoint path: `result/ckpts/ckpts_FAiD_bx_hx`.

## Text to Image Generation

![AR-RAG Showcase](https://github.com/jingyq1/AR-RAG/blob/main/images/quality.png)

### DAiD

```bash
python arrag/t2i_example/t2i_daid_L.sh
```

The default output image path: `result/generated_imgs/example_t2i_daid.jpg`.

### FAiD

```bash
python arrag/t2i_example/t2i_faid_L.sh
```

The default output image path: `result/generated_imgs/example_t2i_faid.jpg`.

<!-- 
## Citation

If you find our work useful for your research, please cite:

```bibtex
@InProceedings{ICCV2025_PATCHRAG,
  author    = {Anonymous ICCV submission},
  title     = {PATCH-RAG: Patch-Based Retrieval Augmentation for Photorealistic Image Generation},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
}
``` -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.
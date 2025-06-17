import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import random
path_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{path_workspace}/arrag/Janus')
from janus.models import VLChatProcessor
from janus.models.janus_retri_conv_ds import MultiModalityCausalLM
from janus.models.llama_conv import SmoothPatchBlender
import random
import datasets
import pdb

# =====================  1) DATASET AND COLLATOR  =====================

@dataclass
class DataCollator:
    vl_chat_processor: VLChatProcessor
    max_text_length: int
    max_image_length: int
    retrieved_key_num: int

    def process_text(self, text: str) -> str:
        conversation = [
            {
                "role": "<|User|>",
                "content": text
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        return sft_format + self.vl_chat_processor.image_start_tag

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Each feature in `features` has:
            {
              "caption.txt": str,
              "code.npy": List[int],
            }
        """

        # ----------------------------------------------------------
        # 1) TEXT TOKENIZATION (batched)
        # ----------------------------------------------------------
        prompts = [self.process_text(f["caption.txt"]) for f in features]
        tok_out = self.vl_chat_processor.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_input_ids = torch.LongTensor(tok_out["input_ids"])  # shape: [B, T_text]

        text_attention_mask = tok_out["attention_mask"]  # shape: [B, T_text]

        batch_size = len(features)

        # ----------------------------------------------------------
        # 2) PREP IMAGE TOKENS (batched)
        #    ground_truth_image_tokens is a list of int (VQ code indices).
        #    We pad them to `max_image_length`.
        # ----------------------------------------------------------
        # Build a tensor [batch_size, max_image_length]
        # Fill with -100 or some pad token. We'll treat them carefully in forward.
        ground_truth_lens = [len(f["code.npy"]) for f in features]
        max_len_img = min(self.max_image_length, max(ground_truth_lens))
        ground_truth_image_tokens = torch.full(
            (batch_size, max_len_img), fill_value=-100, dtype=torch.long
        )

        for i, feat in enumerate(features):
            length = min(len(feat["code.npy"]), max_len_img)
            ground_truth_image_tokens[i, :length] = torch.tensor(
                feat["code.npy"][:length], dtype=torch.long
            )

        # 3) RETRIEVED IMAGE TOKENS (padded)
        #    We'll create shape [B, max_len_img, retrieved_key_num]
        #    - If `feat["retrieved_codes.npy"][j]` is None => fill with -1 (codes) and 0.0 (scores).
        retrieved_codes_tensor = torch.full(
            (batch_size, max_len_img, self.retrieved_key_num),
            fill_value=-1,
            dtype=torch.long
        )

        for i, feat in enumerate(features):
            length = min(len(feat["code.npy"]), max_len_img)
            codes_per_image = feat["retrieved_codes.npy"]   # list/array of length L_img

            for j in range(length):
                if codes_per_image[j] is None:
                    # skip; they stay -1/0.0 in the padded tensor
                    continue

                # Suppose each codes_per_image[j] is a list of length `self.retrieved_key_num`
                # or at least up to it. We'll copy them in, or clamp at K.
                k_len = min(len(codes_per_image[j]), self.retrieved_key_num)
                random.shuffle(codes_per_image[j])
                retrieved_codes_tensor[i, j, :k_len] = torch.tensor(
                    codes_per_image[j][:k_len], dtype=torch.long
                )

        # ----------------------------------------------------------
        # 4) CREATE LABELS (the model will shift inside forward or use huggingface shift).
        #    For standard language modeling, you'd do shift-by-1. But here:
        #    We want to predict the NEXT image token, so the label dimension
        #    can match ground_truth_image_tokens. We'll keep text tokens' labels = -100
        #    so they won't contribute to the loss.
        # ----------------------------------------------------------
        # We'll just combine text + image in forward. For now, store them separately.
        # The model’s forward can do the shift or can do a standard approach. Let’s keep it simple.
        labels = ground_truth_image_tokens.clone()
        # text portion is set to -100. We'll do that in the model or collate. Right now we have them separate,
        # so the model will figure out how to align them. 
        # If needed, we can do: text_label_placeholder = -100 * torch.ones_like(text_input_ids)

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "ground_truth_image_tokens": ground_truth_image_tokens,
            "retrieved_codes": retrieved_codes_tensor,   # [B, max_len_img, K]
            "labels": labels,  # shape [B, max_len_img], for next-token prediction
        }

# =====================  3) LOCAL TRAINER (SIMPLIFIED)  =====================
def freeze_non_lora_parameters(model: nn.Module):
    """
    Freeze all parameters that are NOT inside LoRA layers, plus any custom
    trainable parameter like self.retrieval_alpha, if you want to keep it trainable
    or freeze it as well.
    """
    for n, p in model.named_parameters():
        # If it is a LoRA parameter or you want it tunable, skip freezing.
        if "lora_" in n:
            p.requires_grad = True
        elif 'blender' in n:
            p.requires_grad = True
        elif 'conv_weights' in n:
            p.requires_grad = True
        elif 'retrieval_layernorm' in n:
            p.requires_grad = True
        elif 'gen_head' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Huggingface dataset path/name.")
    parser.add_argument("--model_name_or_path", type=str, default='deepseek-ai/Janus-Pro-1B')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--max_image_length", type=int, default=576)
    parser.add_argument("--retrieved_key_num", type=int, default=10)
    parser.add_argument("--num_blender", type=int, required=True)
    parser.add_argument("--num_hop", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    # Deepspeed will be specified in command line: --deepspeed ds_config.json
    args = parser.parse_args()
    numhop = args.num_hop.replace(',', '')
    args.output_dir = f'{args.output_dir}/ckpts_FAiD_b{args.num_blender}_h{numhop}'
    print(f"OUTPUT_DIR: {args.output_dir}")
    args.num_hop = [int(x) for x in args.num_hop.split(',')]

    # 1) Load dataset
    train_dataset = datasets.load_from_disk(args.train_data)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    print(f"Total training samples: {len(train_dataset)}")

    # 2) Tokenizer
    # Usually the same name as model_name_or_path if it includes a tokenizer
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = vl_chat_processor.tokenizer

    # 3) DataCollator
    data_collator = DataCollator(
        vl_chat_processor=vl_chat_processor,
        max_text_length=args.max_text_length,
        max_image_length=args.max_image_length,
        retrieved_key_num=args.retrieved_key_num,
    )

    # 4) Load model (with `MyImageGenModel` wrapper)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    setattr(config.language_config, 'num_SmoothPatchBlender', 24//args.num_blender)
    setattr(config.language_config, 'num_hop', args.num_hop)
    base_model = MultiModalityCausalLM.from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.bfloat16)

    # Add this block to initialize the blender modules
    for name, module in base_model.named_modules():
        if isinstance(module, SmoothPatchBlender):
            print(f"Initializing blender module: {name}")
            # Explicitly call your initialization method
            module._init_weights()

    base_model.config.use_cache = False  # important for training

    # 5) Setup LoRA on the language model only
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model.language_model, lora_config)
    
    # inject back
    base_model.language_model = lora_model

    # 6) Freeze everything not in LoRA + the custom retrieval_alpha
    freeze_non_lora_parameters(base_model)

    # 7) TrainingArguments + HF Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # or more if desired
        fp16=False,
        bf16=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=9000,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        seed=args.seed,
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8) Train
    # print trainable parameters
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    trainer.train()
    # merge the peft model to the base model
    base_model.language_model.merge_and_unload()
    base_model.language_model = base_model.language_model.base_model.model
    
    
    # 9) Save final
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    

if __name__ == "__main__":
    main()

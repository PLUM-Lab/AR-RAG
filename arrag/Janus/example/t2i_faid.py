import os
import PIL.Image
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from janus.models import VLChatProcessor
from janus.models.janus_retri_conv_ds import MultiModalityCausalLM
import time
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from retrieve.retrieve_L  import FaissRetrieverAllIndexes
import argparse
import json

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    retriever: FaissRetrieverAllIndexes = None,
    retri_num: int = 1,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(mmgpt.gen_embed.weight.device)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id # unconditioned padding
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(mmgpt.gen_embed.weight.device)
    generated_tokens_4retri = torch.full(
            (parallel_size, image_token_num_per_image),
            fill_value=-1,
            dtype=torch.long
        )

    image_dim = img_size // patch_size
    previous_img_hidden_states = None
    for i in range(image_token_num_per_image):
        ## incorporate the retrieved distribition
        if retriever is not None and i>image_dim and (i-1)%image_dim<image_dim-retri_num-1:
            new_tokens_L = torch.cat([generated_tokens[0, i-image_dim-1:i-image_dim+2], generated_tokens[:, i-1]], dim=0)
            _, retrieved_codes = patch_retrieve(new_tokens_L, mmgpt.gen_vision_model, retriever)
            retrieved_codes_clamped = torch.where(
                retrieved_codes < 0,
                torch.zeros_like(retrieved_codes),
                retrieved_codes
            )  # [B, L_img, K]

            # embed => [B, L_img, K, n_embed]
            retrieved_codes_clamped = retrieved_codes_clamped.to(mmgpt.gen_embed.weight.device)
            retrieved_emb = mmgpt.gen_embed(retrieved_codes_clamped)
            # align => [B, L_img, K, hidden_size]
            retrieved_emb_aligned = mmgpt.gen_aligner(retrieved_emb).unsqueeze(0)
        else:
            retrieved_emb_aligned = torch.zeros((1, 1, 10, 2048), dtype=torch.bfloat16).to(mmgpt.gen_embed.weight.device)

        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            retrieved_features=retrieved_emb_aligned,#[:,i,:,:].unsqueeze(1), # [B,S,K,D] [1,1,10,2048]
            pos_id = i,
            previous_img_hidden_states = previous_img_hidden_states,
            use_cache=True, 
            past_key_values=outputs.past_key_values if i != 0 else None,
            # output_hidden_states=True
        )
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        generated_tokens_4retri[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return PIL.Image.fromarray(visual_img[0])

def patch_retrieve(ls_tokens, vq_model, retriever):
    # mask of ls_tokens where -1
    mask = ls_tokens == -1
    ls_tokens = torch.where(
        ls_tokens < 0,
        torch.zeros_like(ls_tokens),
        ls_tokens
    )
    query = vq_model.quantize.get_codebook_entry(ls_tokens)
    query[mask] = 0.0
    query = query.cpu().detach().float().numpy()
    final_distances_t, final_codes_t = retriever.search(query, construct_train=True) # D, I, codes, distribution
    return final_distances_t, final_codes_t

def prepare_prompt(prompt, vl_chat_processor):
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    return prompt

# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)    
    parser.add_argument('--img_caption', type=str, required=True)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--retriever_path', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--retriver_device_id', type=str, default='0')
    args = parser.parse_args()
    args.retriver_device_id = [int(x) for x in args.retriver_device_id.split(',')]

    # initialize the model
    if args.device_id is not None:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-1B")
    
    # initialize the retriever
    assert args.retriever_path is not None, "Retriever path must be provided."
    retriever = FaissRetrieverAllIndexes(
        base_path=args.retriever_path,
        dim=32,
        top_k=10,
        use_gpu=True,
        dim_vocab=16384,
        nprobe=4096,
        device_id=args.retriver_device_id,
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # generate images
    caption = prepare_prompt(args.img_caption, vl_chat_processor)
    
    # generate image
    image = generate(vl_gpt,vl_chat_processor,prompt=caption, retriever=retriever)
    image.save(os.path.join(args.save_dir, "example_t2i_faid.jpg"))

if __name__ == '__main__':
    main()
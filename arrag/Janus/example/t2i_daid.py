import os
import PIL.Image
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from janus.models import MultiModalityCausalLM, VLChatProcessor
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
    retri_ratio: float = 0,
    retri_num: int = 1,
    retri_temperature: float = 1,
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

    image_dim = img_size // patch_size
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        
        probs = torch.softmax(logits / temperature, dim=-1)

        ## incorporate the retrieved distribition
        if retriever is not None and retri_ratio>0 and i>image_dim and (i-1)%image_dim<image_dim-retri_num-1:
            new_tokens_L = torch.cat([generated_tokens[0, i-image_dim-1:i-image_dim+2], generated_tokens[:, i-1]], dim=0)

            query = mmgpt.gen_vision_model.quantize.get_codebook_entry(new_tokens_L).cpu().detach().float().numpy()
            final_distances, final_codes_t = retriever.search(query,unique=True) # D, I, codes, distribution
            distances = torch.tensor(final_distances, device=mmgpt.device).squeeze()
            if len(distances.shape) == 0:
                distances = distances.unsqueeze(0)
            prob_retri = torch.softmax(-distances / retri_temperature, dim=-1)
            prob_retri_full = torch.zeros_like(probs, dtype=prob_retri.dtype)
            prob_retri_full[0, final_codes_t.squeeze()] = prob_retri
            probs = (1 - retri_ratio) * probs + retri_ratio * prob_retri_full

        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().detach().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return PIL.Image.fromarray(visual_img[0])

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
    parser.add_argument('--img_caption', type=str, required=True)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--retriever_path', type=str, default=None)
    parser.add_argument('--retri_ratio', type=float, default=0.0)
    parser.add_argument('--retri_temperature', type=float, default=1.0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--retriver_device_id', type=list, default=[0,1])
    args = parser.parse_args()

    # initialize the model
    if args.device_id is not None:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vl_gpt = MultiModalityCausalLM.from_pretrained("deepseek-ai/Janus-Pro-1B", trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-1B")

    # initialize the retriever
    assert args.retriever_path is not None, "Retriever path must be provided."
    retriever = FaissRetrieverAllIndexes(
        base_path=args.retriever_path,
        dim=32,
        top_k=2048,
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
    image.save(os.path.join(args.save_dir, "example_t2i_daid.jpg"))

if __name__ == '__main__':
    main()
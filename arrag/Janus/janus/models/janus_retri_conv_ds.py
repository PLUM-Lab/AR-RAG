import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector
from janus.models.llama_conv import LlamaForCausalLM
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

import pdb


class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))
    
    def forward(
        self,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.LongTensor,
        ground_truth_image_tokens: torch.LongTensor,
        retrieved_codes: torch.LongTensor,
        labels: torch.LongTensor = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        text_input_ids:  [B, T_text]
        text_attention_mask: [B, T_text]
        ground_truth_image_tokens: [B, L_img]
        retrieved_codes: [B, L_img, K]   (some entries possibly = -1)
        labels: [B, L_img]
        """

        device = text_input_ids.device
        B, T_text = text_input_ids.shape
        _, L_img = ground_truth_image_tokens.shape

        # -----------------------------------------------------------
        # 1) TEXT embeddings (via the language_model’s standard embeddings)
        # -----------------------------------------------------------
        # get text input embeddings
        # mask = torch.rand(text_input_ids.size(0)) < 0.1
        # # Replace tokens from index 1 to -1 with 100002 for rows where mask is True
        # text_input_ids[mask, 1:-1] = 100002
        text_emb = self.language_model.get_input_embeddings()(text_input_ids)
        # shape: [B, T_text, hidden_size]
        # We'll keep the standard LLM forward approach, but we handle the final cat ourselves.

        # -----------------------------------------------------------
        # 2) GROUND TRUTH IMAGE embeddings
        #    tokens -> self.gen_embed -> shape [B, L_img, n_embed]
        #    then -> self.gen_aligner -> shape [B, L_img, hidden_size]
        # -----------------------------------------------------------
        # Replace any negative tokens with 0 just to avoid error in embedding
        gt_img_tokens_clamped = torch.where(
            ground_truth_image_tokens < 0,
            torch.zeros_like(ground_truth_image_tokens),
            ground_truth_image_tokens,
        )
        gt_img_emb = self.gen_embed(gt_img_tokens_clamped)  # shape [B, L_img, n_embed]
        final_img_emb = self.gen_aligner(gt_img_emb)  # shape [B, L_img, hidden_size]

        # -------------------------------------------------------------------------
        # 3) RETRIEVED EMBEDDINGS (all at once, using vector ops)
        #    retrieved_codes: [B, L_img, K], possibly with -1 for "None" entries.
        # -------------------------------------------------------------------------

        # build a mask for retrieved codes = -1
        retrieved_codes_mask = retrieved_codes == -1
        retrieved_codes_clamped = torch.where(
            retrieved_codes < 0,
            torch.zeros_like(retrieved_codes),
            retrieved_codes
        )  # [B, L_img, K]

        # embed => [B, L_img, K, n_embed]
        retrieved_emb = self.gen_embed(retrieved_codes_clamped)
        # align => [B, L_img, K, hidden_size]
        retrieved_emb_aligned = self.gen_aligner(retrieved_emb)
        retrieved_emb_aligned[retrieved_codes_mask,:] = 0.0

        # -----------------------------------------------------------
        # 5) CONCAT text and image embeddings
        #    input_embeds: [B, T_text + L_img, hidden_size]
        #    We also build an attention_mask
        # -----------------------------------------------------------
        input_embeds = torch.cat([text_emb, final_img_emb], dim=1)
        attention_mask = torch.ones(B, T_text + L_img, dtype=text_attention_mask.dtype, device=device)
        # fill in the text part from text_attention_mask
        attention_mask[:, :T_text] = text_attention_mask

        # -----------------------------------------------------------
        # 6) PREPARE THE LABELS for autoregressive prediction
        #    We want to predict the next image token for each position in ground_truth_image_tokens.
        #    The text tokens do not produce a label => -100.
        #    We can shift them if we follow standard HF approach, but let's do it by providing
        #    the entire sequence to the model and letting it handle the shift.
        #    We'll produce a single labels of shape [B, T_text + L_img].
        #      text portion = -100
        #      image portion = ground_truth_image_tokens
        #    Then the LLM will do a standard language modeling objective with a causal shift inside the model.
        # -----------------------------------------------------------
        new_labels = torch.full((B, T_text + L_img), fill_value=-100, dtype=torch.long, device=device)
        new_labels[:, T_text:] = labels  # fill in the image tokens portion

        # -----------------------------------------------------------
        # 7) FORWARD the entire sequence into the LLM
        # -----------------------------------------------------------
        outputs = self.language_model.model.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            retrieved_features=retrieved_emb_aligned,
            use_cache=False, 
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        # 2) Project hidden_states to image-token logits
        logits = self.gen_head(hidden_states)  # shape [B, T, image_token_size]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = new_labels[:, 1:].contiguous()

        # 3) Build your label mask so that text portion is -100, and the image portion is your ground_truth_image_tokens
        #    (or whichever shifting logic you want).
        #    Suppose you have new_labels of shape [B, T], where the text portion is -100 and
        #    new_labels[:, T_text:] = the actual image tokens.

        if shift_labels is not None:
            # standard cross-entropy ignoring index = -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return CausalLMOutput(loss=loss, logits=logits)
        else:
            return CausalLMOutput(loss=None, logits=logits)

AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure.
    Based on Section 3.4 from "Common Diffusion Noise Schedules and Sample Steps are Flawed."

    Args:
        noise_cfg (torch.Tensor): The predicted noise tensor for the guided diffusion process.
        noise_pred_text (torch.Tensor): The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (float, optional): A rescale factor applied to the noise predictions.
            Defaults to 0.0.

    Returns:
        torch.Tensor: The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.
    Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (SchedulerMixin):
            The scheduler to get timesteps from.
        num_inference_steps (int, optional):
            The number of diffusion steps used when generating samples with a pre-trained model.
        device (str or torch.device, optional):
            The device to which the timesteps should be moved to.
        timesteps (List[int], optional):
            Custom timesteps used to override the timestep spacing strategy of the scheduler.
        sigmas (List[float], optional):
            Custom sigmas used to override the timestep spacing strategy of the scheduler.

    Returns:
        Tuple[torch.Tensor, int]: A tuple where the first element is the timestep schedule
        from the scheduler and the second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class AelifStableDiffusionXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    """
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from `DiffusionPipeline`. Check the superclass documentation
    for the generic methods the library implements for all the pipelines (such as downloading
    or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - `TextualInversionLoaderMixin.load_textual_inversion` for loading textual inversion embeddings
        - `FromSingleFileMixin.from_single_file` for loading `.ckpt` files
        - `StableDiffusionXLLoraLoaderMixin.load_lora_weights` for loading LoRA weights
        - `StableDiffusionXLLoraLoaderMixin.save_lora_weights` for saving LoRA weights
        - `IPAdapterMixin.load_ip_adapter` for loading IP Adapters

    Args:
        vae (AutoencoderKL):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder (CLIPTextModel):
            Frozen text-encoder for SDXL. Typically the text portion of CLIP (e.g. clip-vit-large-patch14).
        text_encoder_2 (CLIPTextModelWithProjection):
            Second frozen text-encoder. Typically the text & pool portion of CLIP (e.g. laion/CLIP-ViT-bigG-14-laion2B).
        tokenizer (CLIPTokenizer):
            Tokenizer of class CLIPTokenizer.
        tokenizer_2 (CLIPTokenizer):
            Second tokenizer of class CLIPTokenizer, used for the second text encoder.
        unet (UNet2DConditionModel):
            Conditional U-Net architecture to denoise the encoded image latents.
        scheduler (KarrasDiffusionSchedulers):
            A scheduler to be used in combination with unet to denoise the encoded image latents.
        image_encoder (CLIPVisionModelWithProjection, optional):
            Optional image encoder (e.g. CLIP Vision model).
        feature_extractor (CLIPImageProcessor, optional):
            Optional image processor (feature extractor).
        force_zeros_for_empty_prompt (bool, optional, defaults to True):
            Whether negative prompts shall default to zero if `negative_prompt` is not explicitly provided.
        add_watermarker (bool, optional):
            Whether to use the invisible_watermark library to watermark output images.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

        # The scale factor for the VAE
        if getattr(self, "vae", None) is not None:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # The default image size for sampling
        if hasattr(self, "unet") and self.unet is not None and hasattr(self.unet.config, "sample_size"):
            self.default_sample_size = self.unet.config.sample_size
        else:
            self.default_sample_size = 128

        if add_watermarker is None:
            add_watermarker = is_invisible_watermark_available()
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    ############################################################################
    # AELIF-RELATED METHODS (MIRRORING THE SD3-STYLE IMPLEMENTATION)
    ############################################################################

    @staticmethod
    def elementwise_multiply_embeddings_with_noise(
        tensor: torch.Tensor,
        percentage: float,
        generator: Optional[torch.Generator] = None,
        mean: float = 0.0,
        std: float = 1.0
    ) -> torch.Tensor:
        """
        Multiplies a subset of tokens with noise drawn from N(mean, std).
        Excludes the first and last embeddings (e.g. special tokens).
        """
        if tensor.dim() != 3:
            raise ValueError("Tensor must be of shape [batch_size, num_tokens, embedding_dim]")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1")

        device = tensor.device
        batch_size, num_tokens, embedding_dim = tensor.shape

        if num_tokens < 3:
            raise ValueError("Tensor must have at least 3 tokens to exclude first and last embeddings.")

        start = 1
        end = num_tokens - 1
        num_tokens_in_interval = end - start
        if num_tokens_in_interval <= 0:
            raise ValueError("Invalid token interval: no tokens available after first/last exclusion.")

        num_tokens_to_modify = int(num_tokens_in_interval * percentage)
        num_tokens_to_modify = min(num_tokens_to_modify, num_tokens_in_interval)
        if num_tokens_to_modify == 0:
            return tensor.clone()

        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(0)
        elif generator.device != torch.device("cpu"):
            raise ValueError("Generator must be on 'cpu' device.")

        available_indices = torch.arange(start, end, device="cpu").unsqueeze(0).repeat(batch_size, 1)
        shuffled_indices = available_indices.gather(
            1,
            torch.randperm(num_tokens_in_interval, generator=generator).unsqueeze(0).repeat(batch_size, 1)
        )
        selected_indices = shuffled_indices[:, :num_tokens_to_modify].to(device)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_tokens_to_modify)

        noise = (torch.randn(batch_size, num_tokens_to_modify, embedding_dim, generator=generator) * std + mean).to(device)
        modified_tensor = tensor.clone()
        modified_tensor[batch_indices, selected_indices, :] *= noise

        return modified_tensor

    @staticmethod
    def mask_embeddings_per_sample(
        tensor: torch.Tensor,
        percentage: float,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Zeroes out a subset of middle tokens (excluding first and last).
        """
        if tensor.dim() != 3:
            raise ValueError("Tensor must be of shape [batch_size, num_tokens, embedding_dim]")
        if not (0 <= percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1")

        device = tensor.device
        batch_size, num_tokens, embedding_dim = tensor.shape

        if num_tokens < 3:
            raise ValueError("Tensor must have at least 3 tokens to exclude first and last embeddings.")

        start = 1
        end = num_tokens - 1
        num_tokens_in_interval = end - start
        if num_tokens_in_interval <= 0:
            raise ValueError("No tokens to mask after excluding first/last.")

        num_tokens_to_mask = int(num_tokens_in_interval * percentage)
        num_tokens_to_mask = min(num_tokens_to_mask, num_tokens_in_interval)
        if num_tokens_to_mask == 0:
            return tensor.clone()

        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(0)
        elif generator.device != torch.device("cpu"):
            raise ValueError("Generator must be on 'cpu' device.")

        available_indices = torch.arange(start, end, device="cpu").unsqueeze(0).repeat(batch_size, 1)
        shuffled_indices = available_indices.gather(
            1,
            torch.randperm(num_tokens_in_interval, generator=generator).unsqueeze(0).repeat(batch_size, 1)
        )
        selected_indices = shuffled_indices[:, :num_tokens_to_mask].to(device)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_tokens_to_mask)

        mask = torch.ones(batch_size, num_tokens, embedding_dim, device=device)
        mask[batch_indices, selected_indices, :] = 0
        masked_tensor = tensor * mask

        return masked_tensor

    def AELIF_augmentation(
        self,
        text_embeddings: torch.Tensor,
        aelif_function_name: Callable,
        percentage: float,
        std_aelif: float,
        mean_aelif: float,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Applies either masking or noise-based augmentation to a subset of tokens
        (excluding first/last) in `text_embeddings`.
        """
        text_embeddings_copy = copy.deepcopy(text_embeddings)

        if 'mask' in aelif_function_name.__name__:
            augmented_embeddings = aelif_function_name(
                text_embeddings_copy,
                percentage=percentage,
                generator=generator
            )
        else:
            augmented_embeddings = aelif_function_name(
                text_embeddings_copy,
                percentage=percentage,
                generator=generator,
                mean=mean_aelif,
                std=std_aelif,
            )

        return augmented_embeddings

    #########################################################################
    # MAIN PROMPT ENCODING + PIPELINE LOGIC
    #########################################################################

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        # ---------- AELIF additions ----------
        aelif: Optional[str] = None,
        aelif_percentage: float = 0.0,
        std_aelif: float = 1.0,
        mean_aelif: float = 0.0,
        generator: Optional[torch.Generator] = None,
        # -------------------------------------
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Extended to optionally apply AELIF augmentations to token embeddings.

        Args:
            prompt (str or List[str]):
                The prompt or prompts to guide the image generation.
            prompt_2 (str or List[str], optional):
                A second prompt or prompts used for the second text encoder.
            device (torch.device, optional):
                Which device to use for inference.
            num_images_per_prompt (int, optional):
                How many images to generate per prompt.
            do_classifier_free_guidance (bool, optional):
                Whether to employ classifier-free guidance (CFG).
            negative_prompt (str or List[str], optional):
                Negative prompt to dissuade certain styles or objects.
            negative_prompt_2 (str or List[str], optional):
                Second negative prompt for the second text encoder.
            prompt_embeds (torch.Tensor, optional):
                Pre-generated text embeddings.
            negative_prompt_embeds (torch.Tensor, optional):
                Pre-generated negative text embeddings.
            pooled_prompt_embeds (torch.Tensor, optional):
                Pooled embeddings for the positive prompt (text_encoder_2).
            negative_pooled_prompt_embeds (torch.Tensor, optional):
                Pooled embeddings for the negative prompt (text_encoder_2).
            lora_scale (float, optional):
                Scaling factor for LoRA if loaded.
            clip_skip (int, optional):
                How many layers to skip from the end of the text encoder's stack.
            aelif (str, optional):
                If set to "mask" or "noise_conv", applies that augmentation.
                If None, no augmentation is applied.
            aelif_percentage (float, optional):
                Fraction of middle tokens to augment, from 0.0 to 1.0.
            std_aelif (float, optional):
                Std dev for noise-based augmentation.
            mean_aelif (float, optional):
                Mean for noise-based augmentation.
            generator (torch.Generator, optional):
                Optional RNG for reproducibility.

        Returns:
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        # define tokenizers + text encoders
        if self.tokenizer is not None:
            tokenizers = [self.tokenizer, self.tokenizer_2]
            text_encoders = [self.text_encoder, self.text_encoder_2]
        else:
            tokenizers = [self.tokenizer_2]
            text_encoders = [self.text_encoder_2]

        # ------------------------------------------------------
        # PROMPT ENCODING
        # ------------------------------------------------------
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            if isinstance(prompt_2, str):
                prompt_2 = [prompt_2]

            prompt_embeds_list = []
            prompts = [prompt, prompt_2]

            for p, tok, txt_enc in zip(prompts, tokenizers, text_encoders):
                # textual inversion
                if isinstance(self, TextualInversionLoaderMixin):
                    p = self.maybe_convert_prompt(p, tok)

                text_inputs = tok(
                    p,
                    padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tok(p, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = tok.batch_decode(untruncated_ids[:, tok.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tok.model_max_length} tokens: {removed_text}"
                    )

                p_embeds = txt_enc(text_input_ids.to(device), output_hidden_states=True)

                if pooled_prompt_embeds is None and p_embeds[0].ndim == 2:
                    pooled_prompt_embeds = p_embeds[0]

                if clip_skip is None:
                    p_embeds = p_embeds.hidden_states[-2]
                else:
                    p_embeds = p_embeds.hidden_states[-(clip_skip + 2)]

                # apply AELIF on prompts if requested
                if aelif is not None:
                    if aelif == "noise_conv":
                        p_embeds = self.AELIF_augmentation(
                            p_embeds,
                            self.elementwise_multiply_embeddings_with_noise,
                            aelif_percentage,
                            std_aelif,
                            mean_aelif,
                            generator=generator,
                        )
                    elif aelif == "mask":
                        p_embeds = self.AELIF_augmentation(
                            p_embeds,
                            self.mask_embeddings_per_sample,
                            aelif_percentage,
                            0.0,
                            0.0,
                            generator=generator,
                        )

                prompt_embeds_list.append(p_embeds)

            # combine from text_encoder_1 and text_encoder_2
            prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)

        # ------------------------------------------------------
        # NEGATIVE PROMPT ENCODING
        # ------------------------------------------------------
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt

        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            # quick zero-out
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            if isinstance(negative_prompt_2, str):
                negative_prompt_2 = [negative_prompt_2]

            uncond_tokens = [negative_prompt, negative_prompt_2]
            negative_prompt_embeds_list = []

            for nprompt, tok, txt_enc in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    nprompt = self.maybe_convert_prompt(nprompt, tok)

                max_length = prompt_embeds.shape[1]
                uncond_input = tok(
                    nprompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                np_embeds = txt_enc(uncond_input.input_ids.to(device), output_hidden_states=True)

                if negative_pooled_prompt_embeds is None and np_embeds[0].ndim == 2:
                    negative_pooled_prompt_embeds = np_embeds[0]
                np_embeds = np_embeds.hidden_states[-2]

                # apply AELIF for negative if user wants
                if aelif is not None:
                    if aelif == "noise_conv":
                        np_embeds = self.AELIF_augmentation(
                            np_embeds,
                            self.elementwise_multiply_embeddings_with_noise,
                            aelif_percentage,
                            std_aelif,
                            mean_aelif,
                            generator=generator,
                        )
                    elif aelif == "mask":
                        np_embeds = self.AELIF_augmentation(
                            np_embeds,
                            self.mask_embeddings_per_sample,
                            aelif_percentage,
                            0.0,
                            0.0,
                            generator=generator,
                        )

                negative_prompt_embeds_list.append(np_embeds)

            negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)

        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        """
        Encodes an image using the image encoder, optionally returning hidden states for IP-Adapter usage.
        """
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)

        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)
            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self,
        ip_adapter_image,
        ip_adapter_image_embeds,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance
    ):
        """
        Prepares image embeddings for IP Adapter usage if provided,
        either by encoding an input image or using pre-generated embeddings.
        """
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []

        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have the same length as the number of IP Adapters. "
                    f"Got {len(ip_adapter_image)} images but {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds_result = []
        for i, single_image_embeds in enumerate(image_embeds):
            repeated_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                repeated_neg_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                repeated_embeds = torch.cat([repeated_neg_embeds, repeated_embeds], dim=0)

            repeated_embeds = repeated_embeds.to(device=device)
            ip_adapter_image_embeds_result.append(repeated_embeds)

        return ip_adapter_image_embeds_result

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        Prepare extra kwargs for the scheduler step, since not all schedulers
        have the same signature. Eta corresponds to eta in the DDIM paper.
        """
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """
        Validates the pipeline inputs to ensure all are consistent and
        no illegal combinations or shapes are provided.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8 but are {height}, {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` must be a positive integer, got {callback_steps} of type {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None:
            if not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` must be in {self._callback_tensor_inputs}, but found invalid keys."
                )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` must be `str` or `list` but is {type(prompt)}.")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` must be `str` or `list` but is {type(prompt_2)}.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`: {negative_prompt_embeds}."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape. "
                    f"Got {prompt_embeds.shape} vs {negative_prompt_embeds.shape}."
                )
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` must also be passed."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` must also be passed."
            )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`, not both."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` must be a `list` but is {type(ip_adapter_image_embeds)}."
                )
            elif ip_adapter_image_embeds and ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    "`ip_adapter_image_embeds` must be a list of 3D or 4D tensors."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None
    ):
        """
        Prepares the latent variables for the denoising process.
        """
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"List of generators has length {len(generator)}, but requested effective batch size {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None
    ):
        """
        Constructs the time-ids or additional conditioning used by SDXL, 
        for micro-conditioning techniques such as resizing/cropping hints.
        """
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but {passed_add_embed_dim} was created."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def upcast_vae(self):
        """
        Ensures the VAE is in float32 mode if needed (to avoid overflow in float16),
        while preserving memory via xformers or torch 2.0 attn if available.
        """
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Builds a guidance-scale embedding vector used for time_cond_proj in some SD models.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        # --- AELIF add-ins (optional) ---
        aelif: Optional[str] = None,
        aelif_percentage: float = 0.0,
        std_aelif: float = 1.0,
        mean_aelif: float = 0.0,
        generator: Optional[torch.Generator] = None,
        # --------------------------------
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Main text-to-image call for SDXL, with optional AELIF text-embedding augmentation.

        Args:
            aelif (str, optional):
                The augmentation method ("mask" or "noise_conv"). If None, no augmentation is applied.
            aelif_percentage (float, optional):
                The fraction of middle tokens to augment. 0.0 <= x <= 1.0.
            std_aelif (float, optional):
                Std dev for noise-based augmentation.
            mean_aelif (float, optional):
                Mean for noise-based augmentation.
            generator (torch.Generator, optional):
                Optional random seed generator, must be on CPU if used for augmentation.

            prompt (str or List[str], optional):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            prompt_2 (str or List[str], optional):
                A second prompt for text_encoder_2 if desired. If None, it uses `prompt`.
            height (int, optional):
                The height in pixels of the generated image. Defaults to 1024 for best results.
            width (int, optional):
                The width in pixels of the generated image. Defaults to 1024 for best results.
            num_inference_steps (int, optional):
                The number of denoising steps. More steps -> better quality, but slower generation.
            timesteps (List[int], optional):
                Custom timesteps for schedulers that support them.
            sigmas (List[float], optional):
                Custom sigmas for schedulers that support them.
            denoising_end (float, optional):
                Fraction of the total denoising process to complete before stopping. If < 1, stops early.
            guidance_scale (float, optional):
                CFG scale, typically between 1.0 and 20.0. Higher -> more adherence to prompt.
            negative_prompt (str or List[str], optional):
                The prompt or prompts not to guide the image generation.
            negative_prompt_2 (str or List[str], optional):
                A second negative prompt if desired (for the second text encoder).
            num_images_per_prompt (int, optional):
                Number of images to generate per prompt.
            eta (float, optional):
                The eta parameter for DDIM. 0.0 is deterministic.
            latents (torch.Tensor, optional):
                Pre-generated noisy latents. If not provided, a random sample is used.
            prompt_embeds (torch.Tensor, optional):
                Pre-generated text embeddings for the positive prompt.
            negative_prompt_embeds (torch.Tensor, optional):
                Pre-generated text embeddings for the negative prompt.
            pooled_prompt_embeds (torch.Tensor, optional):
                Pooled embeddings for the positive prompt (text_encoder_2).
            negative_pooled_prompt_embeds (torch.Tensor, optional):
                Pooled embeddings for the negative prompt (text_encoder_2).
            ip_adapter_image (PipelineImageInput, optional):
                Image input for IP-Adapter usage.
            ip_adapter_image_embeds (List[torch.Tensor], optional):
                Pre-generated IP-Adapter embeddings.
            output_type (str, optional):
                The output format ("pil" or "numpy"). Defaults to "pil".
            return_dict (bool, optional):
                Whether to return a `StableDiffusionXLPipelineOutput` or a tuple.
            cross_attention_kwargs (Dict[str, Any], optional):
                Extra kwargs for cross-attention operations.
            guidance_rescale (float, optional):
                Factor to rescale the CFG strength mid-generation, helps with overexposure.
            original_size (Tuple[int, int], optional):
                Micro-conditioning original size hints for SDXL. Defaults to (width, height).
            crops_coords_top_left (Tuple[int, int], optional):
                Micro-conditioning crop coordinates.
            target_size (Tuple[int, int], optional):
                Micro-conditioning target size hints for SDXL. Defaults to (width, height).
            negative_original_size (Tuple[int, int], optional):
                Negative version of `original_size`.
            negative_crops_coords_top_left (Tuple[int, int], optional):
                Negative version of `crops_coords_top_left`.
            negative_target_size (Tuple[int, int], optional):
                Negative version of `target_size`.
            clip_skip (int, optional):
                If set, skip the last `clip_skip` layers in the text encoder.
            callback_on_step_end (Callable or PipelineCallback or MultiPipelineCallbacks, optional):
                A function or callback that is called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (List[str], optional):
                Which tensor inputs to pass to `callback_on_step_end`.
            kwargs:
                Other custom arguments.

        Returns:
            StableDiffusionXLPipelineOutput or tuple:
            If return_dict=True, returns `StableDiffusionXLPipelineOutput(images=...)`.
            Otherwise, returns a tuple of images.
            
        Examples:
                ```
                ```
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, please use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, please use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default the height/width if not provided
        if height is None:
            height = self.default_sample_size * self.vae_scale_factor
        if width is None:
            width = self.default_sample_size * self.vae_scale_factor

        if original_size is None:
            original_size = (height, width)
        if target_size is None:
            target_size = (height, width)

        # 1. Check input correctness
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # store some states
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Determine batch size
        if prompt is not None:
            if isinstance(prompt, str):
                batch_size = 1
            else:
                batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompt (AELIF is applied here if aelif is set)
        lora_scale = self.cross_attention_kwargs["scale"] if (self.cross_attention_kwargs and "scale" in self.cross_attention_kwargs) else None
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self._clip_skip,
            aelif=aelif,
            aelif_percentage=aelif_percentage,
            std_aelif=std_aelif,
            mean_aelif=mean_aelif,
            generator=generator,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare add_time_ids
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # IP-Adapter logic if provided
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        else:
            image_embeds = None

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 handle denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and 0 < self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len([ts for ts in timesteps if ts >= discrete_timestep_cutoff])
            timesteps = timesteps[:num_inference_steps]

        # optionally get guidance scale embedding if needed
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }
                if image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self._cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                    # old style callback
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        # convert latents to images
        if output_type != "latent":
            needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.config.force_upcast)
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    self.vae = self.vae.to(latents.dtype)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None

            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if output_type != "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

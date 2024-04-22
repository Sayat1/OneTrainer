from contextlib import nullcontext
from uuid import uuid4

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionDepth2ImgPipeline, \
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation

from modules.model.BaseModel import BaseModel
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.convert.rescale_noise_scheduler_to_zero_terminal_snr import \
    rescale_noise_scheduler_to_zero_terminal_snr
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class StableDiffusionModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_vector: Tensor | None,
            placeholder: str,
    ):
        token_count = text_encoder_vector.shape[0]

        self.uuid = uuid
        self.text_encoder_vector = text_encoder_vector
        self.placeholder = placeholder
        self.text_tokens = [f"<{uuid4()}>" for _ in range(token_count)]


class StableDiffusionModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer: CLIPTokenizer
    noise_scheduler: DDIMScheduler
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    image_depth_processor: DPTImageProcessor
    depth_estimator: DPTForDepthEstimation

    # autocast context
    autocast_context: torch.autocast | nullcontext

    train_dtype: DataType

    # persistent embedding training data
    embedding: StableDiffusionModelEmbedding | None
    embedding_state: Tensor | None
    additional_embeddings: list[StableDiffusionModelEmbedding] | None
    additional_embedding_states: list[Tensor | None]
    embedding_wrapper: AdditionalEmbeddingWrapper

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    sd_config: dict | None
    sd_config_filename: str | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer: CLIPTokenizer | None = None,
            noise_scheduler: DDIMScheduler | None = None,
            text_encoder: CLIPTextModel | None = None,
            vae: AutoencoderKL | None = None,
            unet: UNet2DConditionModel | None = None,
            image_depth_processor: DPTImageProcessor | None = None,
            depth_estimator: DPTForDepthEstimation | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embedding: StableDiffusionModelEmbedding | None = None,
            embedding_state: Tensor | None = None,
            additional_embeddings: list[StableDiffusionModelEmbedding] | None = None,
            additional_embedding_states: list[Tensor | None] = None,
            embedding_wrapper: AdditionalEmbeddingWrapper | None = None,
            text_encoder_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
            lora_state_dict: dict | None = None,
            sd_config: dict | None = None,
            sd_config_filename: str | None = None,
            model_spec: ModelSpec | None = None
    ):
        super(StableDiffusionModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec
        )

        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.image_depth_processor = image_depth_processor
        self.depth_estimator = depth_estimator

        self.autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32

        self.embedding = embedding
        self.embedding_state = embedding_state
        self.additional_embeddings = additional_embeddings if additional_embeddings is not None else []
        self.additional_embedding_states = additional_embedding_states if additional_embedding_states is not None else []
        self.embedding_wrapper = embedding_wrapper

        self.text_encoder_lora = text_encoder_lora
        self.unet_lora = unet_lora
        self.lora_state_dict = lora_state_dict

        self.sd_config = sd_config
        self.sd_config_filename = sd_config_filename

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def depth_estimator_to(self, device: torch.device):
        if self.depth_estimator is not None:
            self.depth_estimator.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def unet_to(self, device: torch.device):
        self.unet.to(device=device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.depth_estimator_to(device)
        self.text_encoder_to(device)
        self.unet_to(device)

    def eval(self):
        self.vae.eval()
        if self.depth_estimator is not None:
            self.depth_estimator.eval()
        self.text_encoder.eval()
        self.unet.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        if self.model_type.has_depth_input():
            return StableDiffusionDepth2ImgPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                depth_estimator=self.depth_estimator,
                feature_extractor=self.image_depth_processor,
            )
        elif self.model_type.has_conditioning_image_input():
            return StableDiffusionInpaintPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        else:
            return StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        
    def create_pipeline_without_unet(self)  -> DiffusionPipeline:
        return StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=None,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
    
    def create_pipeline_unet(self)  -> DiffusionPipeline:
        return StableDiffusionPipeline(
                vae=None,
                text_encoder=None,
                tokenizer=None,
                unet=self.unet,
                scheduler=None,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )

    def force_v_prediction(self):
        self.noise_scheduler.config.prediction_type = 'v_prediction'
        self.sd_config['model']['params']['parameterization'] = 'v'

    def force_epsilon_prediction(self):
        self.noise_scheduler.config.prediction_type = 'epsilon'
        self.sd_config['model']['params']['parameterization'] = 'epsilon'

    def rescale_noise_scheduler_to_zero_terminal_snr(self):
        rescale_noise_scheduler_to_zero_terminal_snr(self.noise_scheduler)

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        for embedding in self.additional_embeddings:
            embedding_string = ''.join(embedding.text_tokens)
            prompt = prompt.replace(embedding.placeholder, embedding_string)

        if self.embedding is not None:
            embedding_string = ''.join(self.embedding.text_tokens)
            prompt = prompt.replace(self.embedding.placeholder, embedding_string)

        return prompt

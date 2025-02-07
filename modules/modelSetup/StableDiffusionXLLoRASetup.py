from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from lycoris import create_lycoris, LycorisNetwork,create_lycoris_from_weights
import torch
from itertools import chain

class StableDiffusionXLLoRASetup(
    BaseStableDiffusionXLSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_1_lora",
                parameters=model.text_encoder_1_lora.parameters(),
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.text_encoder_2.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_2_lora",
                parameters=model.text_encoder_2_lora.parameters(),
                learning_rate=config.text_encoder_2.learning_rate,
            ))

        if config.train_any_embedding():
            if config.text_encoder.train_embedding:
                self._add_embedding_param_groups(
                    model.embedding_wrapper_1, parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_1"
                )

            if config.text_encoder_2.train_embedding:
                self._add_embedding_param_groups(
                    model.embedding_wrapper_2, parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_2"
                )

        if config.unet.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="unet_lora",
                parameters=model.unet_lora.parameters(),
                learning_rate=config.unet.learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = config.text_encoder.train and \
                                   not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]

            train_embedding_1 = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_1_vector.requires_grad_(train_embedding_1)

            train_embedding_2 = \
                embedding_config.train \
                and config.text_encoder_2.train_embedding \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_2_vector.requires_grad_(train_embedding_2)

        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = config.text_encoder_2.train and \
                                   not self.stop_text_encoder_2_training_elapsed(config, model.train_progress)
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)

        if model.unet_lora is not None:
            train_unet = config.unet.train and \
                         not self.stop_unet_training_elapsed(config, model.train_progress)
            model.unet_lora.requires_grad_(train_unet)

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        create_te1 = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te1")
        create_te2 = config.text_encoder_2.train or state_dict_has_prefix(model.lora_state_dict, "lora_te2")
        create_unet = config.unet.train or state_dict_has_prefix(model.lora_state_dict, "lora_unet")

        if model.lora_state_dict:
            LycorisNetwork.LORA_PREFIX = "lora_te1"
            model.text_encoder_1_lora,_ = create_lycoris_from_weights(1.0, None, model.text_encoder_1, weights_sd=model.lora_state_dict) if create_te1 else None

            LycorisNetwork.LORA_PREFIX = "lora_te2"
            model.text_encoder_2_lora,_ = create_lycoris_from_weights(1.0, None, model.text_encoder_2, weights_sd=model.lora_state_dict) if create_te2 else None

            LycorisNetwork.LORA_PREFIX = "lora_unet"
            model.unet_lora,_ = create_lycoris_from_weights(1.0, None, model.unet, weights_sd=model.lora_state_dict)
            
            model.lora_state_dict = None
        else:
            LycorisNetwork.LORA_PREFIX = "lora_te1"
            model.text_encoder_1_lora = create_lycoris(model.text_encoder_1, 1.0, linear_dim=config.lora_rank, linear_alpha=config.lora_alpha, algo=str(config.lora_type), dropout=config.dropout_probability, **config.lycoris_options) if create_te1 else None

            LycorisNetwork.LORA_PREFIX = "lora_te2"
            model.text_encoder_2_lora = create_lycoris(model.text_encoder_2, 1.0, linear_dim=config.lora_rank, linear_alpha=config.lora_alpha, algo=str(config.lora_type), dropout=config.dropout_probability, **config.lycoris_options) if create_te2 else None

            LycorisNetwork.LORA_PREFIX = "lora_unet"
            model.unet_lora = create_lycoris(model.unet, 1.0, linear_dim=config.lora_rank, linear_alpha=config.lora_alpha, algo=str(config.lora_type), dropout=config.dropout_probability, **config.lycoris_options)

        if create_te1:
            model.text_encoder_1_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_1_lora.apply_to()
        if create_te2:
            model.text_encoder_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_2_lora.apply_to()

        model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.unet_lora.apply_to()

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        if config.do_edm_style_training:
            from modules.util import create
            from modules.util.enum.NoiseScheduler import NoiseScheduler
            model.noise_scheduler = create.create_noise_scheduler(NoiseScheduler.EULER, model.noise_scheduler, 1000)

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, self.create_parameters(model, config),train_device=self.train_device)

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        vae_on_train_device = config.align_prop or not config.latent_caching
        text_encoder_1_on_train_device = \
            config.train_text_encoder_or_embedding()\
            or config.align_prop \
            or not config.latent_caching
        text_encoder_2_on_train_device = \
            config.train_text_encoder_2_or_embedding() \
            or config.align_prop \
            or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if config.text_encoder.train:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if config.text_encoder_2.train:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.eval()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()


    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.embedding_wrapper_1.normalize_embeddings()
            model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)

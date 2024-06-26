import torch

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.NamedParameterGroup import NamedParameterGroupCollection, NamedParameterGroup
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import init_model_parameters


class StableDiffusionLoRASetup(
    BaseStableDiffusionSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionLoRASetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            if config.lora_te_separate_train:
                ex_layer_name=""
                params=[]
                te_index= 0
                for key,module in model.text_encoder_lora.lora_modules.items():
                    layer_name = ".".join(key.split(".")[:4])
                    if ex_layer_name != layer_name:
                        if len(params) > 0:
                            parameter_group_collection.add_group(NamedParameterGroup(
                                unique_name=f"text_encoder_lora{te_index}",
                                display_name=f"text_encoder_lora{te_index}",
                                parameters=params,
                                learning_rate=config.text_encoder.learning_rate,
                            ))
                        params=[]
                        te_index += 1
                        ex_layer_name = layer_name
                    params.extend(module.parameters())
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"text_encoder_lora{te_index}",
                    display_name=f"text_encoder_lora{te_index}",
                    parameters=params,
                    learning_rate=config.text_encoder.learning_rate,
                ))
            else:
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name="text_encoder_lora",
                    display_name="text_encoder_lora",
                    parameters=model.text_encoder_lora.parameters(),
                    learning_rate=config.text_encoder.learning_rate,
                ))

        if config.train_any_embedding():
            for parameter, placeholder, name in zip(model.embedding_wrapper.additional_embeddings,
                                                    model.embedding_wrapper.additional_embedding_placeholders,
                                                    model.embedding_wrapper.additional_embedding_names):
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"embeddings/{name}",
                    display_name=f"embeddings/{placeholder}",
                    parameters=[parameter],
                    learning_rate=config.embedding_learning_rate,
                ))

        if config.unet.train:
            if config.lora_unet_separate_train:
                for i,module_pair in enumerate(model.unet_lora.block_parameters().items()):
                    params=[]
                    key = module_pair[0]
                    moudles = module_pair[1]
                    for module in moudles:
                        params.extend(module.parameters())
                    parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"unet_lora{i}",
                    display_name=f"unet_lora{i}",
                    parameters=params,
                    learning_rate=config.unet.learning_rate,
                    ))
            else:
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name="unet_lora",
                    display_name="unet_lora",
                    parameters=model.unet_lora.parameters(),
                    learning_rate=config.unet.learning_rate,
                ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_lora is not None:
            train_text_encoder = config.text_encoder.train and \
                                 not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_lora.requires_grad_(train_text_encoder)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = embedding_config.train and \
                              not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_vector.requires_grad_(train_embedding)

        if model.unet_lora is not None:
            train_unet = config.unet.train and \
                         not self.stop_unet_training_elapsed(config, model.train_progress)
            model.unet_lora.requires_grad_(train_unet)

    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        if config.text_encoder.train:
            model.text_encoder_lora = LoRAModuleWrapper(
                model.text_encoder, config.lora_rank, "lora_te", config.lora_alpha, dora_wd=config.lora_dora_wd
            )

        model.unet_lora = LoRAModuleWrapper(
            model.unet, config.lora_rank, "lora_unet", config.lora_alpha, config.lora_modules, config.lora_conv_rank, config.lora_conv_alpha, config.lora_rank_ratio, config.lora_alpha_ratio, config.lora_train_blocks, dora_wd=config.lora_dora_wd
        )

        if model.lora_state_dict:
            if config.text_encoder.train:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.unet_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if config.text_encoder.train:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
        model.unet_lora.set_dropout(config.dropout_probability)

        if config.text_encoder.train:
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())

        if config.text_encoder.train:
            model.text_encoder_lora.hook_to_module()
        model.unet_lora.hook_to_module()

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config))

        self._setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop or not config.latent_caching
        text_encoder_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)
        model.depth_estimator_to(self.temp_device)

        if config.text_encoder.train:
            model.text_encoder.train()
        else:
            model.text_encoder.eval()

        model.vae.eval()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)

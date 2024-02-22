from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionLoRASetup(BaseStableDiffusionSetup):
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
    ) -> Iterable[Parameter]:
        params = list()

        if config.train_text_encoder:
            params += list(model.text_encoder_lora.parameters())

        if config.train_unet:
            params += list(model.unet_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()
        

        if config.train_text_encoder:
            if config.lora_te_separate_train:
                ex_layer_name=""
                params=[]
                for key,module in model.text_encoder_lora.modules.items():
                    layer_name = ".".join(key.split(".")[:4])
                    if ex_layer_name != layer_name:
                        if len(params) > 0:
                            param_groups.append(
                                self.create_param_groups(config, params, config.text_encoder_learning_rate)
                            )
                        params=[]
                        ex_layer_name = layer_name
                    params.extend(module.parameters())
                param_groups.append(
                    self.create_param_groups(config, params, config.text_encoder_learning_rate)
                )
            else:
                param_groups.append(
                    self.create_param_groups(config, model.text_encoder_lora.parameters(), config.text_encoder_learning_rate)
                )
            

        if config.train_unet:
            if config.lora_unet_separate_train:
                for key,modules in model.unet_lora.block_parameters().items():
                    params=[]
                    for module in modules:
                        params.extend(module.parameters())
                    param_groups.append(
                        self.create_param_groups(config, params, config.unet_learning_rate)
                    )
            else:
                param_groups.append(
                self.create_param_groups(config, model.unet_lora.parameters(), config.unet_learning_rate)
                )

        return param_groups

    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        if model.text_encoder_lora is None and config.train_text_encoder:
            model.text_encoder_lora = LoRAModuleWrapper(
                model.text_encoder, config.lora_rank, "lora_te", config.lora_alpha
            )

        if model.unet_lora is None:
            model.unet_lora = LoRAModuleWrapper(
                model.unet, config.lora_rank, "lora_unet", config.lora_alpha, config.lora_modules, config.lora_conv_rank, config.lora_conv_alpha, config.lora_rank_ratio, config.lora_alpha_ratio, config.lora_train_blocks
            )

        model.text_encoder.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        if config.train_text_encoder:
            train_text_encoder = config.train_text_encoder and (model.train_progress.epoch < config.train_text_encoder_epochs)
            model.text_encoder_lora.requires_grad_(train_text_encoder)
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        train_unet = config.train_unet and (model.train_progress.epoch < config.train_unet_epochs)
        model.unet_lora.requires_grad_(train_unet)
        model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.unet_lora.hook_to_module()

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, config), model.optimizer_state_dict, config
        )
        del model.optimizer_state_dict

        model.ema = create.create_ema(
            self.create_parameters(model, config), model.ema_state_dict, config
        )
        del model.ema_state_dict

        self.setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop
        text_encoder_on_train_device = config.train_text_encoder or config.align_prop or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)
        model.depth_estimator_to(self.temp_device)

        if config.train_text_encoder:
            model.text_encoder.train()
        else:
            model.text_encoder.eval()

        model.vae.eval()

        if config.train_unet:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.train_text_encoder:
            train_text_encoder = config.train_text_encoder and (model.train_progress.epoch < config.train_text_encoder_epochs)
            model.text_encoder_lora.requires_grad_(train_text_encoder)

        train_unet = config.train_unet and (model.train_progress.epoch < config.train_unet_epochs)
        model.unet_lora.requires_grad_(train_unet)

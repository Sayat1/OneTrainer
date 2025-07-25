import contextlib
import copy
import json
import os
import shutil
import traceback
import psutil
from collections.abc import Callable
from pathlib import Path

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.memory_util import TorchMemoryRecorder
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
from requests.exceptions import ConnectionError
from tqdm import tqdm


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        import random
        import numpy as np
        if config.seed is None or config.seed == -1:
            config.seed = random.randint(0, 2**32)
        print(f"set seed:{config.seed}")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}"))
        if config.tensorboard:
            super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False

        self.grad_hook_handles = []

    def start(self):
        self.__save_config_to_workspace()

        if self.config.clear_cache_before_training and self.config.latent_caching:
            self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(
                    token = self.config.secrets.huggingface_token,
                    new_session = False,
                )

        self.callbacks.on_update_status("loading the model")
        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()
        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model.train_progress, is_validation=True
            )

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text']):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.abspath(os.path.join(self.config.workspace_dir, "backup"))
        print("__prune_backups")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                 os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )
            try:
                for dirpath in backup_directories[backups_to_keep:]:
                    dirpath = os.path.join(backup_dirpath, dirpath)
                    print(f"Deleting old backup {str(dirpath)}")
                    path_dir = Path(dirpath)
                    newpath_dir = path_dir.rename(f"/content/drive/MyDrive/{path_dir.name}/")
                    old_backup_files = newpath_dir.rglob("*.*")
                    for old_file in old_backup_files:
                        if Path(old_file).is_file():
                            path_util.collab_delete_file(old_file)
                    shutil.rmtree(newpath_dir)
            except Exception as e:
                print(f"Could not delete old rolling backup {dirpath}")

        return None

    def __prune_saves(self,saves_to_keep: int,save_path: str,save_base_name: str,save_file_ext: str):
        save_dirpath = Path(save_path).parent
        if save_dirpath.exists():
            save_files_path = sorted(save_dirpath.glob(f"{save_base_name}*{save_file_ext}"),key=os.path.getmtime,reverse=True)
            for save_file_path in save_files_path[saves_to_keep:]:
                try:
                    print(f"Deleting old save {str(save_file_path)}")
                    with open(str(save_file_path),'w') as f:
                        pass
                    os.remove(str(save_file_path))
                    yaml_file = save_dirpath/f"{save_file_path.stem}.yaml"
                    if yaml_file.exists():
                        with open(str(yaml_file),'w') as f:
                            pass
                        os.remove(str(yaml_file))
                except Exception as e:
                    print(f"Could not delete old rolling save {save_file_path}")
        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_config_list: list[SampleConfig],
            folder_postfix: str = "",
            is_custom_sample: bool = False,
    ):
        for i, sample_config in enumerate(sample_config_list):
            if sample_config.enabled:
                try:
                    safe_prompt = path_util.safe_filename(sample_config.prompt)

                    if is_custom_sample:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            "custom",
                        )
                    else:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            f"{str(i)} - {safe_prompt}{folder_postfix}",
                        )

                    sample_path = os.path.join(
                        sample_dir,
                        f"{self.config.save_filename_prefix}{get_string_timestamp()}-training-sample-{train_progress.filename_string()}"
                    )

                    def on_sample_default(sampler_output: ModelSamplerOutput):
                        if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                            self.tensorboard.add_image(
                                f"sample{str(i)} - {safe_prompt}", pil_to_tensor(sampler_output.data),  # noqa: B023
                                train_progress.global_step
                            )
                        self.callbacks.on_sample_default(sampler_output)

                    def on_sample_custom(sampler_output: ModelSamplerOutput):
                        self.callbacks.on_sample_custom(sampler_output)

                    on_sample = on_sample_custom if is_custom_sample else on_sample_default
                    on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                    if psutil.virtual_memory()[1] > 6e+9:
                        self.model.to(self.temp_device)
                    self.model.eval()

                    sample_config = copy.copy(sample_config)
                    sample_config.from_train_config(self.config)

                    self.model_sampler.sample(
                        sample_config=sample_config,
                        destination=sample_path,
                        image_format=self.config.sample_image_format,
                        video_format=self.config.sample_video_format,
                        audio_format=self.config.sample_audio_format,
                        on_sample=on_sample,
                        on_update_progress=on_update_progress,
                    )
                except Exception:
                    traceback.print_exc()
                    print("Error during sampling, proceeding without sampling")

                torch_gc()

    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("sampling")

        is_custom_sample = False
        if not sample_params_list:
            if self.config.samples is not None:
                sample_params_list = self.config.samples
            else:
                with open(self.config.sample_definition_file_name, 'r') as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values().from_dict(samples[i])
                    sample_params_list = samples
        else:
            is_custom_sample = True

        if self.model.ema:
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            is_custom_sample=is_custom_sample,
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if an ema model exists
        if self.model.ema and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                folder_postfix=" - no-ema",
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation)

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress, deterministic=True)
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config)

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                                            average_loss,
                                            train_progress.global_step)

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar("loss/validation_step/total_average",
                                            total_average_loss,
                                            train_progress.global_step)

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.abspath(os.path.join(self.config.workspace_dir, "backup", backup_name))

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("saving")
        output_model_destination_path = Path(self.config.output_model_destination )
        save_path = str(output_model_destination_path.parent / f"{self.config.save_filename_prefix}{output_model_destination_path.stem}-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}")
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
            if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)
            if self.config.rolling_save_count > 0:
                self.__prune_saves(self.config.rolling_save_count,save_path,output_model_destination_path.stem,self.config.output_model_format.file_extension())

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        if self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating fused_back_pass with gradient_accumulation_steps > 1 does not reduce VRAM usage.")

            for param_group in self.model.optimizer.param_groups:
                for i, parameter in enumerate(param_group["params"]):
                    # TODO: Find a better check instead of "parameter.requires_grad".
                    #       This will break if the some parameters don't require grad during the first training step.
                    if parameter.requires_grad:
                        if scaler:
                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                if self.__is_update_step(self.model.train_progress):
                                    scaler.unscale_parameter_(tensor, self.model.optimizer)
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                                    scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                                    tensor.grad = None
                        else:
                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                if self.__is_update_step(self.model.train_progress):
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                                    self.model.optimizer.step_parameter(tensor, param_group, i)
                                    tensor.grad = None

                        handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                        self.grad_hook_handles.append(handle)

    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            self.callbacks.on_update_status("caching")
            for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1),position=0,file=sys.stdout, desc="epoch"):
                self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = 0.0
        ema_loss = None
        ema_loss_steps = 0
        for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1),position=0,file=sys.stdout,leave=True ,desc="epoch"):
            print("")
            self.callbacks.on_update_status("starting epoch/caching")

            if self.config.latent_caching:
                self.data_loader.get_data_set().start_next_epoch()
                self.model_setup.setup_train_device(self.model, self.config)
            else:
                self.model_setup.setup_train_device(self.model, self.config)
                self.data_loader.get_data_set().start_next_epoch()

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            lr_schedulers = []
            if self.config.text_encoder.train:
                lr_schedulers.append(self.config.te1_learning_rate_scheduler if self.config.te1_learning_rate_scheduler!=None else self.config.learning_rate_scheduler)
            if self.config.text_encoder_2.train:
                lr_schedulers.append(self.config.te2_learning_rate_scheduler if self.config.te2_learning_rate_scheduler!=None else self.config.learning_rate_scheduler)
            if self.config.unet.train:
                lr_schedulers.append(self.config.learning_rate_scheduler)
            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_schedulers=lr_schedulers,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step,
                    eta_min=self.config.learning_rate_min,
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()
            step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step",position=0, file=sys.stdout, total=current_epoch_length,
                             initial=train_progress.epoch_step)
            for batch in step_tqdm:
                if self.__needs_sample(train_progress) or self.commands.get_and_reset_sample_default_command():
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )

                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    transferred_to_temp_device = False

                    if self.commands.get_and_reset_backup_command():
                        self.model.to(self.temp_device)
                        self.backup(train_progress, True, step_tqdm.write)
                        transferred_to_temp_device = True

                    if self.commands.get_and_reset_save_command():
                        self.model.to(self.temp_device)
                        self.save(train_progress, True, step_tqdm.write)
                        transferred_to_temp_device = True

                    if transferred_to_temp_device:
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("training")

                with TorchMemoryRecorder(enabled=False):
                    prior_pred_indices = [i for i in range(self.config.batch_size)
                                          if ConceptType(batch['concept_type'][i]) == ConceptType.PRIOR_PREDICTION]
                    if len(prior_pred_indices) > 0 \
                            or (self.config.masked_training
                                and self.config.masked_prior_preservation_weight > 0
                                and self.config.training_method == TrainingMethod.LORA):
                        with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                            #do NOT create a subbatch using the indices, even though it would be more efficient:
                            #different timesteps are used for a smaller subbatch by predict(), but the conditioning must match exactly:
                            prior_model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                        prior_model_prediction = prior_model_output_data['predicted'].to(dtype=model_output_data['target'].dtype)
                        model_output_data['target'][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
                        model_output_data['prior_target'] = prior_model_prediction
                    else:
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                    loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)

                    loss = loss / self.config.gradient_accumulation_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    has_gradient = True
                    accumulated_loss += loss.item()

                    if self.__is_update_step(train_progress):
                        if scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False

                        reported_lr = self.model_setup.report_to_tensorboard(
                            self.model, self.config, lr_scheduler, self.tensorboard
                        )

                        self.tensorboard.add_scalar("loss/train_step", accumulated_loss, train_progress.global_step)
                        ema_loss = ema_loss or accumulated_loss
                        ema_loss_steps += 1
                        ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                        ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss * (1 - ema_loss_decay))
                        update_reports = {
                            'loss': accumulated_loss,
                            'smooth loss': ema_loss,
                        }
                        update_reports.update(reported_lr)
                        step_tqdm.set_postfix(update_reports)
                        self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)
                        accumulated_loss = 0.0

                        self.model_setup.after_optimizer_step(self.model, self.config, train_progress)
                        if self.model.ema:
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step
                            )
                            self.model.ema.step(
                                self.parameters,
                                update_step
                            )

                        self.one_step_trained = True

                if self.config.validation:
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    return

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.one_step_trained:
            self.model.to(self.temp_device)

            if self.config.backup_before_save:
                self.backup(self.model.train_progress)
            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free or self.config.use_schedulefree_wrapper:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            self.callbacks.on_update_status("saving the final model")

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=False)
            if os.path.isdir(self.config.output_model_destination) and self.config.output_model_format.is_single_file():
                save_path = os.path.join(
                    self.config.output_model_destination,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}"
                )
            else:
                save_path = self.config.output_model_destination
            print("Saving " + save_path)

            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )

        self.model.to(self.temp_device)

            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)
                dest_path = Path(self.config.output_model_destination)
                non_ema_dest_path = str(dest_path.parent/f"{dest_path.stem}-non_ema{dest_path.suffix}")
                print("Saving non-ema " + non_ema_dest_path)

                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=non_ema_dest_path,
                    dtype=self.config.output_dtype.torch_dtype()
                )

        self.tensorboard.close()

        if self.config.tensorboard:
            super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()

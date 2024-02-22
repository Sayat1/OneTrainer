import math
from abc import ABCMeta

import torch
from torch import nn, Tensor
from torch.nn import Linear, Conv2d, Parameter
import re

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")
NUM_OF_BLOCKS=12
def get_block_index(lora_name: str) -> int:
    block_idx = -1  # invalid lora name

    m = RE_UPDOWN.search(lora_name)
    if m:
        g = m.groups()
        i = int(g[1])
        j = int(g[3])
        if g[2] == "resnets":
            idx = 3 * i + j
        elif g[2] == "attentions":
            idx = 3 * i + j
        elif g[2] == "upsamplers" or g[2] == "downsamplers":
            idx = 3 * i + 2

        if g[0] == "down":
            block_idx = 1 + idx  # 0に該当するLoRAは存在しない
        elif g[0] == "up":
            block_idx = NUM_OF_BLOCKS + 1 + idx

    elif "mid_block_" in lora_name:
        block_idx = NUM_OF_BLOCKS  # idx=12

    return block_idx

class LoRAModule(metaclass=ABCMeta):
    prefix: str
    orig_module: nn.Module
    lora_down: nn.Module
    lora_up: nn.Module
    alpha: torch.Tensor

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super(LoRAModule, self).__init__()
        self.prefix = prefix.replace('.', '_')
        self.orig_module = orig_module
        self.rank = rank
        self.alpha = torch.tensor(alpha)
        if orig_module is not None:
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

        self.is_applied = False
        self.orig_forward = self.orig_module.forward if self.orig_module is not None else None

    def forward(self, x, *args, **kwargs):
        return self.orig_forward(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)

    def requires_grad_(self, requires_grad: bool):
        self.lora_down.requires_grad_(requires_grad)
        self.lora_up.requires_grad_(requires_grad)

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModule':
        self.lora_down.to(device, dtype)
        self.lora_up.to(device, dtype)
        self.alpha.to(device, dtype)
        return self

    def parameters(self) -> list[Parameter]:
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())

    def load_state_dict(self, state_dict: dict):
        down_state_dict = {
            "weight": state_dict.pop(self.prefix + ".lora_down.weight")
        }
        up_state_dict = {
            "weight": state_dict.pop(self.prefix + ".lora_up.weight")
        }
        self.alpha = state_dict.pop(self.prefix + ".alpha")

        self.lora_down.load_state_dict(down_state_dict)
        self.lora_up.load_state_dict(up_state_dict)

    def state_dict(self) -> dict:
        state_dict = {}
        state_dict[self.prefix + ".lora_down.weight"] = self.lora_down.weight.data
        state_dict[self.prefix + ".lora_up.weight"] = self.lora_up.weight.data
        state_dict[self.prefix + ".alpha"] = self.alpha
        return state_dict

    def hook_to_module(self):
        if not self.is_applied:
            self.orig_module.forward = self.forward
            self.is_applied = True

    def remove_hook_from_module(self):
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.is_applied = False

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass


class LinearLoRAModule(LoRAModule):
    def __init__(self, prefix: str, orig_module: Linear, rank: int, alpha: float, rank_ratio: float, alpha_ratio: float):
        in_features = orig_module.in_features
        out_features = orig_module.out_features
        my_rank = rank
        my_alpha = alpha
        if rank_ratio > 0.0:
            my_rank = int(min(in_features,out_features) * rank_ratio)
            my_alpha = my_rank * alpha_ratio
        super(LinearLoRAModule, self).__init__(prefix, orig_module, my_rank, my_alpha)


        self.lora_down = Linear(in_features, my_rank, bias=False, device=orig_module.weight.device)
        self.lora_up = Linear(my_rank, out_features, bias=False, device=orig_module.weight.device)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class Conv2dLoRAModule(LoRAModule):
    def __init__(self, prefix: str, orig_module: Conv2d, rank: int, alpha: float, rank_ratio: float, alpha_ratio: float):
        in_channels = orig_module.in_channels
        out_channels = orig_module.out_channels
        my_rank = rank
        my_alpha = alpha
        if rank_ratio > 0.0:
            my_rank = int(min(in_channels,out_channels) * rank_ratio)
            my_alpha = my_rank * alpha_ratio
        super(Conv2dLoRAModule, self).__init__(prefix, orig_module, my_rank, my_alpha)
        
        kernel_size = orig_module.kernel_size
        stride = orig_module.stride
        padding = orig_module.padding

        self.lora_down = Conv2d(in_channels, my_rank, kernel_size,stride,padding, bias=False, device=orig_module.weight.device)
        self.lora_up = Conv2d(my_rank, out_channels, (1, 1), (1, 1), bias=False, device=orig_module.weight.device)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class DummyLoRAModule(LoRAModule):
    def __init__(self, prefix: str):
        super(DummyLoRAModule, self).__init__(prefix, None, 1, 1)
        self.lora_down = None
        self.lora_up = None

        self.save_state_dict = {}

    def requires_grad_(self, requires_grad: bool):
        pass

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModule':
        pass

    def parameters(self) -> list[Parameter]:
        return []

    def load_state_dict(self, state_dict: dict):
        self.save_state_dict = {
            self.prefix + ".lora_down.weight": state_dict.pop(self.prefix + ".lora_down.weight"),
            self.prefix + ".lora_up.weight": state_dict.pop(self.prefix + ".lora_up.weight"),
            self.prefix + ".alpha": state_dict.pop(self.prefix + ".alpha"),
        }

    def state_dict(self) -> dict:
        return self.save_state_dict

    def hook_to_module(self):
        pass

    def remove_hook_from_module(self):
        pass

    def apply_to_module(self):
        pass

    def extract_from_module(self, base_module: nn.Module):
        pass


class LoRAModuleWrapper:
    orig_module: nn.Module
    rank: int

    modules: dict[str, LoRAModule]

    def __init__(
            self,
            orig_module: nn.Module | None,
            rank: int,
            prefix: str,
            alpha: float = 1.0,
            module_filter: list[str] = None,
            conv_rank: int = 0,
            conv_alpha: float  = 0.0,
            rank_ratio: float = 0.0,
            alpha_ratio: float = 0.0,
            train_blocks: list[int] = None
    ):
        super(LoRAModuleWrapper, self).__init__()
        self.orig_module = orig_module
        self.prefix = prefix
        self.module_filter = module_filter if module_filter is not None else []
        if conv_rank>0 and conv_alpha<=0:
            conv_alpha = conv_rank
        if rank_ratio>0.0 and alpha_ratio<=0.0:
            alpha_ratio = 1.0
        if train_blocks != None:
            print("BLOCK TRAINING")
            if len(train_blocks)!=25:
                raise KeyError("train_blocks must have 25 numbers")
            if not all(block==1 or block==0 for block in train_blocks):
                raise KeyError("train_blocks must have the value 0 or 1")

        self.modules = self.__create_modules(orig_module, rank,alpha, conv_rank, conv_alpha, rank_ratio, alpha_ratio,train_blocks)

    def __create_modules(self, orig_module: nn.Module | None,rank:int, alpha: float, conv_rank:int, conv_alpha:float, rank_ratio:float, alpha_ratio:float ,train_blocks:list[int]) -> dict[str, LoRAModule]:
        lora_modules = {}

        if orig_module is not None:
            for name, module in orig_module.named_modules():             
                if len(self.module_filter) == 0 or any([x in name for x in self.module_filter]):
                    training_block=True
                    if train_blocks:
                        lora_name = self.prefix + "_" + name
                        lora_name = lora_name.replace(".", "_") 
                        block_idx = get_block_index(lora_name)
                        if block_idx != -1:
                            training_block = train_blocks[block_idx] == 1
                    if training_block:
                        if isinstance(module, Linear):
                            print(self.prefix + "_" + name)
                            lora_modules[name] = LinearLoRAModule(self.prefix + "_" + name, module, rank, alpha, rank_ratio, alpha_ratio)
                        elif isinstance(module, Conv2d):
                            print(self.prefix + "_" + name)
                            if module.kernel_size == (1,1):
                                lora_modules[name] = Conv2dLoRAModule(self.prefix + "_" + name, module, rank, alpha, rank_ratio, alpha_ratio)
                            elif conv_rank > 0:
                                lora_modules[name] = Conv2dLoRAModule(self.prefix + "_" + name, module, conv_rank, conv_alpha, rank_ratio, alpha_ratio)

        return lora_modules

    def requires_grad_(self, requires_grad: bool):
        for name, module in self.modules.items():
            module.requires_grad_(requires_grad)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for name, module in self.modules.items():
            parameters += module.parameters()
        return parameters

    def block_parameters(self) -> dict:
        block_idx_to_lora = {}
        for name, module in self.modules.items():
            idx = get_block_index(name.replace(".", "_") )
            if idx not in block_idx_to_lora:
                block_idx_to_lora[idx] = []
            block_idx_to_lora[idx].append(module)
        return block_idx_to_lora

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModuleWrapper':
        for name, module in self.modules.items():
            module.to(device, dtype)
        return self

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        """
        Loads the state dict

        Args:
            state_dict: the state dict
        """

        # create a copy, so the modules can pop states
        state_dict = {k: v for (k, v) in state_dict.items() if k.startswith(self.prefix)}

        for name, module in self.modules.items():
            module.load_state_dict(state_dict)

        # create dummy modules for the remaining keys
        remaining_names = list(state_dict.keys())
        for name in remaining_names:
            if name.endswith(".alpha"):
                prefix = name.removesuffix(".alpha")
                module = DummyLoRAModule(prefix)
                module.load_state_dict(state_dict)
                self.modules[prefix] = module

    def state_dict(self) -> dict:
        """
        Returns the state dict
        """
        state_dict = {}

        for name, module in self.modules.items():
            state_dict |= module.state_dict()

        return state_dict

    def hook_to_module(self):
        """
        Hooks the LoRA into the module without changing its weights
        """
        for name, module in self.modules.items():
            module.hook_to_module()

    def remove_hook_from_module(self):
        """
        Removes the LoRA hook from the module without changing its weights
        """
        for name, module in self.modules.items():
            module.remove_hook_from_module()

    def apply_to_module(self):
        """
        Applys the LoRA to the module, changing its weights
        """
        for name, module in self.modules.items():
            module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """
        Creates a LoRA from the difference between the base_module and the orig_module
        """
        for name, module in self.modules.items():
            module.extract_from_module(base_module)

    def prune(self):
        """
        Removes all dummy modules
        """
        self.modules = {k: v for (k, v) in self.modules.items() if not isinstance(v, DummyLoRAModule)}

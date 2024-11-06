from enum import Enum

import torch


class Optimizer(Enum):
    # Sorted by origin (BNB / torch first, then DADAPT), then by adapter name, then interleaved by variant.

    # BNB Standard & 8-bit
    ADAGRAD = 'ADAGRAD'
    ADAGRAD_8BIT = 'ADAGRAD_8BIT'

    # 32 bit is torch and not bnb
    ADAM = 'ADAM'
    ADAM_8BIT = 'ADAM_8BIT'

    # 32 bit is torch and not bnb
    ADAMW = 'ADAMW'
    ADAMW_8BIT = 'ADAMW_8BIT'

    AdEMAMix = 'AdEMAMix'
    AdEMAMix_8BIT = "AdEMAMix_8BIT"

    LAMB = 'LAMB'
    LAMB_8BIT = 'LAMB_8BIT'

    LARS = 'LARS'
    LARS_8BIT = 'LARS_8BIT'

    LION = 'LION'
    LION_8BIT = 'LION_8BIT'

    RMSPROP = 'RMSPROP'
    RMSPROP_8BIT = 'RMSPROP_8BIT'

    # 32 bit is torch and not bnb
    SGD = 'SGD'
    SGD_8BIT = 'SGD_8BIT'

    # Schedule-free optimizers
    SCHEDULE_FREE_ADAMW = 'SCHEDULE_FREE_ADAMW'
    SCHEDULE_FREE_SGD = 'SCHEDULE_FREE_SGD'

    # DADAPT
    DADAPT_ADA_GRAD = 'DADAPT_ADA_GRAD'
    DADAPT_ADAM = 'DADAPT_ADAM'
    DADAPT_ADAN = 'DADAPT_ADAN'
    DADAPT_LION = 'DADAPT_LION'
    DADAPT_SGD = 'DADAPT_SGD'

    # Prodigy
    PRODIGY = 'PRODIGY'

    # ADAFACTOR
    ADAFACTOR = 'ADAFACTOR'

    # CAME
    CAME = 'CAME'

    #Pytorch Optimizers
    ADABELIEF = 'ADABELIEF'
    TIGER = 'TIGER'
    AIDA = 'AIDA'
    SOAP = 'SOAP'

    @property
    def is_adaptive(self):
        return self in [
            self.DADAPT_SGD,
            self.DADAPT_ADAM,
            self.DADAPT_ADAN,
            self.DADAPT_ADA_GRAD,
            self.DADAPT_LION,
            self.PRODIGY,
        ]

    @property
    def is_schedule_free(self):
        return self in [
            self.SCHEDULE_FREE_ADAMW,
            self.SCHEDULE_FREE_SGD,
        ]

    def supports_fused_back_pass(self):
        return self in [
            Optimizer.ADAFACTOR,
            Optimizer.CAME,
            Optimizer.ADAM,
            Optimizer.ADAMW,
        ]

    # Small helper for adjusting learning rates to adaptive optimizers.
    def maybe_adjust_lrs(self, lrs: dict[str, float], optimizer: torch.optim.Optimizer):
        dlrs= lrs.copy()
        if self.is_adaptive:
            for i,item in enumerate(lrs.items()):
                if "dlr" in optimizer.param_groups[i]:
                    dlrs.update({f"dlr[{i}]":optimizer.param_groups[i]["dlr"]})
                else:
                    dlrs.update({f"dlr[{i}]":float(item[1])*optimizer.param_groups[i]["d"]})
            return dlrs
        elif self.is_schedule_free or "schedulefree" in type(optimizer).__name__.lower():
            for i,item in enumerate(lrs.items()):
                if "lr_max" in optimizer.param_groups[i]:
                    dlrs.update({f"lr_max[{i}]":optimizer.param_groups[i]["lr_max"]})
                else:
                    dlrs.update({item[0]:float(item[1])})
            return dlrs
        if "mecha" in type(optimizer).__name__.lower():
            s = optimizer.state['_mechanic']['s']
            s_sum = torch.sum(s).item()
            for i,item in enumerate(lrs.items()):
                dlrs.update({f"s*lr[{i}]":float(item[1])*s_sum})
            return dlrs
        return lrs

    def __str__(self):
        return self.value

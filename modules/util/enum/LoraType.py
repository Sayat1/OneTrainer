from enum import Enum


class LoraType(Enum):
    LORA = 'lora'
    LOHA = 'loha'
    LOKR = 'lokr'
    FULL = "full"
    IA3 = "ia3"
    DYLORA = "dylora"
    GLORA = "glora"
    DIAGOFT= "diag-oft"
    BOFT = "boft"

    def __str__(self):
        return self.value

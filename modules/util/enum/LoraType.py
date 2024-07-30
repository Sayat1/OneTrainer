from enum import Enum


class LoraType(Enum):
    LORA = 'LORA'
    LOHA = 'LOHA'
    LOKR = 'LOKR'
    FULL = "FULL"
    IA3 = "IA3"
    DYLORA = "DYLORA"
    GLORA = "GLORA"
    DIAGOFT= "DIAGOFT-oft"
    BOFT = "BOFT"

    def __str__(self):
        return self.value

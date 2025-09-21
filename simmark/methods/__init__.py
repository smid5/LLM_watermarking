from .expmin import ExpMinProcessor, expmin_detect
from .unigram import UnigramProcessor, unigram_detect
from .soft_red_list import SoftRedProcessor, softred_detect
from .synthid import SynthIDProcessor, synthid_detect
from .nomark import NoMarkProcessor
from .watermax_fixed import WaterMaxProcessor, watermax_detect


logit_processors = {
    "expmin": ExpMinProcessor,
    "unigram": UnigramProcessor,
    "softred": SoftRedProcessor,
    "synthid": SynthIDProcessor,
    "nomark": NoMarkProcessor,
    "watermax": WaterMaxProcessor
}

detection_methods = {
    "expmin": expmin_detect,
    "unigram": unigram_detect,
    "softred": softred_detect,
    "synthid": synthid_detect,
    "watermax": watermax_detect
}
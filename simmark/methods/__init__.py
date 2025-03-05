from .expmin import ExpMinProcessor, expmin_detect
from .unigram import UnigramProcessor, unigram_detect
from .soft_red_list import SoftRedProcessor, softred_detect
from .simmark import SimMarkProcessor, simmark_detect
from .synthid import SynthIDProcessor, synthid_detect
from .nomark import NoMarkProcessor

logit_processors = {
    "expmin": ExpMinProcessor,
    "unigram": UnigramProcessor,
    "softred": SoftRedProcessor,
    "simmark": SimMarkProcessor,
    "synthid": SynthIDProcessor,
    "nomark": NoMarkProcessor,
}

detection_methods = {
    "expmin": expmin_detect,
    "unigram": unigram_detect,
    "softred": softred_detect,
    "simmark": simmark_detect,
    "synthid": synthid_detect
}
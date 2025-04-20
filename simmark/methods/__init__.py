from .expmin import ExpMinProcessor, expmin_detect
from .unigram import UnigramProcessor, unigram_detect
from .soft_red_list import SoftRedProcessor, softred_detect
from .simmark import SimMarkProcessor, simmark_detect
from .synthid import SynthIDProcessor, synthid_detect
from .expmin_nohash import ExpMinNoHashProcessor, expmin_nohash_detect
from .nomark import NoMarkProcessor

logit_processors = {
    "expmin": ExpMinProcessor,
    "expminnohash": ExpMinNoHashProcessor,
    "unigram": UnigramProcessor,
    "softred": SoftRedProcessor,
    "simmark": SimMarkProcessor,
    "synthid": SynthIDProcessor,
    "nomark": NoMarkProcessor
}

detection_methods = {
    "expmin": expmin_detect,
    "expminnohash": expmin_nohash_detect,
    "unigram": unigram_detect,
    "softred": softred_detect,
    "simmark": simmark_detect,
    "synthid": synthid_detect
}
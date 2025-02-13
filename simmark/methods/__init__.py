from .expmin import ExpMinProcessor, expmin_detect
from .redgreen import RedGreenProcessor, redgreen_detect
from .simmark import SimMarkProcessor, simmark_detect
from .nomark import NoMarkProcessor

logit_processors = {
    "expmin": ExpMinProcessor,
    "redgreen": RedGreenProcessor,
    "simmark": SimMarkProcessor,
    "nomark": NoMarkProcessor,
}

detection_methods = {
    "expmin": expmin_detect,
    "redgreen": redgreen_detect,
    "simmark": simmark_detect,
}
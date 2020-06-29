from abc import ABC
import copy

from .TRPO import TRPO, TRPO_CONFIG
from agents.NPG import NPG_Gaussian


class TRPO_Gaussian(TRPO, NPG_Gaussian, ABC):
    def __init__(self, parameters):
        config = copy.deepcopy(TRPO_CONFIG)
        config.update(parameters)
        super(TRPO_Gaussian, self).__init__(config)

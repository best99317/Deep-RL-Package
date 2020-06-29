from abc import ABC

import torch
import copy
import numpy as np

from .TRPO import TRPO, TRPO_CONFIG
from agents.NPG import NPG_Softmax


class TRPO_Softmax(TRPO, NPG_Softmax, ABC):
    def __init__(self, parameters):
        config = copy.deepcopy(TRPO_CONFIG)
        config.update(parameters)
        super(TRPO_Softmax, self).__init__(config)


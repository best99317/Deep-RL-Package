import torch
import numpy as np
from agents.HTRPO.HTRPO_Gaussian import HTRPO_Gaussian
from agents.ReFER import ReFER, ReFER2


class ReFER_Gaussian(ReFER2, HTRPO_Gaussian):
    def __init__(self, parameters):
        super(ReFER_Gaussian, self).__init__(parameters)
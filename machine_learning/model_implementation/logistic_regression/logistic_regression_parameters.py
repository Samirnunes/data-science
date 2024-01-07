import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from base_classes.parameters import Parameters

class LogisticRegressionParameters(Parameters):
    def __init__(self):
        self.initial_weights = []
        self.initial_bias = 0
        self.epochs = 100
        self.batch_size = 10
        self.alpha = 0.1
        self.lambda_reg = 0.1
        self.random_state = 0
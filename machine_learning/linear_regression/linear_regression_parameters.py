import sys
sys.path.append('../')
from models.parameters import Parameters

class LinearRegressionParameters(Parameters):
    def __init__(self):
        self.initial_weights = []
        self.initial_bias = 0
        self.epochs = 100
        self.batch_size = 10
        self.alpha = 0.1
        self.lambda_reg = 0.1
        self.random_state = 0
        self.ws = []
        self.b = 0
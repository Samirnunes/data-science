from machine_learning.model_implementation.base_classes.parameters import Parameters


class LogisticRegressionParameters(Parameters):
    def __init__(self):
        self.initial_weights = []
        self.initial_bias = 0
        self.epochs = 100
        self.batch_size = 10
        self.alpha = 0.1
        self.lambda_reg = 0.1
        self.random_state = 0
from abc import ABC, abstractmethod


class Parameters(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __str__(self):
        string = ""
        for key, value in sorted(vars(self).items()):
            string += f'{key} = {value}\n'
        return string

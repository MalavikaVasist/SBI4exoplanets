from abc import ABC, abstractmethod

class Train(ABC):
    @abstractmethod
    def pipe(self, instrument_data, return_loss = True):
        pass



from abc import ABC, abstractmethod

class Worker(ABC):

    @abstractmethod
    def work(self):
        pass

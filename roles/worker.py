from abc import ABC, abstractmethod

class Worker(ABC):

    @abstractmethod
    def work():
        pass

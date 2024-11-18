from mpi4py import MPI
from roles.worker import Worker
from typing import override

class Classifier(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm

    @override
    def work(self):
        pass
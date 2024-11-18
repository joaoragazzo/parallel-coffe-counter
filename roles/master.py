from mpi4py import MPI
from worker import Worker
from typing import override

class Master(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm

    @override
    def work():
        pass
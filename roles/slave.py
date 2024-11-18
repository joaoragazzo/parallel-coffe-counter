from mpi4py import MPI
from worker import Worker
from typing import override

class Slave(Worker):
    def __init__(self, comm: MPI.Comm, **kargs):
        self.comm = comm

    @override
    def work():
        pass
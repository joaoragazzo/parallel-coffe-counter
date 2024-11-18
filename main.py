from mpi4py import MPI
import numpy as np
from roles.master import Master
from roles.slave import Slave
from roles.classifier import Classifier


ROLE_MAP = {
    0: Master,
    1: Classifier
}

def get_process_role(rank, comm, **kwargs):
    role_class = ROLE_MAP.get(rank, Slave) 
    return role_class(comm, **kwargs)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

process = get_process_role(rank, comm)
process.work()



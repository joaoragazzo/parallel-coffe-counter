from mpi4py import MPI
from roles.worker import Worker
from typing import override
import cv2
import tensorflow as tf
import numpy as np
import os


class Classifier(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm

    @override
    def work(self):
        pass

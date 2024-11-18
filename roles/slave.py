from mpi4py import MPI
from roles.worker import Worker
from typing import override

class Slave(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm

    @override
    def work(self):

        while True:
            status = MPI.Status()
            data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            print(data)
            if status.tag == 99:
                # Sinal de parada
                break
            else:
                quadrant_name, quadrant_data = data

                # Processa o quadrante
                # result = classify_coffee_beans(quadrant_data)

                # Envia o resultado de volta ao mestre
                # self.comm.send({quadrant_name: result}, dest=0, tag=22)

        pass
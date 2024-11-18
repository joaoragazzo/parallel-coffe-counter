from mpi4py import MPI
import cv2, math
from roles.worker import Worker
from typing import override

class Master(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm
        self.cap = cv2.VideoCapture(0)
        self.size = comm.Get_size()

    def divide_image(self, frame):
        height, width, _ = frame.shape
        mid_height = height // 2
        mid_width = width // 2

        quadrants = [
            ('superior_esquerdo', frame[0:mid_height, 0:mid_width]),
            ('superior_direito', frame[0:mid_height, mid_width:width]),
            ('inferior_esquerdo', frame[mid_height:height, 0:mid_width]),
            ('inferior_direito', frame[mid_height:height, mid_width:width])
        ]
        return quadrants

    def webcam(self):
        print("Oi")

        if not self.cap.isOpened():
            print("Não foi possível abrir a webcam.")
            self.comm.Abort()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Falha ao capturar o frame.")
                    break

                # Divide a imagem em quadrantes
                quadrants = self.divide_image(frame)

                # Envia os quadrantes para os processos escravos (processos 2 a 5)
                for idx, i in enumerate(range(2, 6)):
                    quadrant_name, quadrant_data = quadrants[idx]
                    # Envia o nome e o dado do quadrante
                    self.comm.send((quadrant_name, quadrant_data), dest=i, tag=11)

                # Recebe os resultados (se necessário)
                # results = {}
                # for i in range(2, 6):
                #     data = self.comm.recv(source=i, tag=22)
                #     results.update(data)

                # Apresenta os resultados (se necessário)
                # print("Resultados da Classificação:")
                # for quadrant_name, result in results.items():
                #     print(f"{quadrant_name}: {result}")

                # Exibe o frame (opcional)
                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupção pelo usuário.")

        finally:
            # Envia sinal de parada aos escravos (se necessário)
            # for i in range(2, 6):
            #     self.comm.send(None, dest=i, tag=99)
            self.cap.release()
            cv2.destroyAllWindows()





        

    @override
    def work(self):
        self.webcam()
        pass
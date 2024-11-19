from mpi4py import MPI
import cv2
import math
from roles.worker import Worker
from typing import override
import pickle  # Import necessário para serialização

class Master(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm
        self.cap = cv2.VideoCapture(2)
        self.size = comm.Get_size()
        self.n = int(math.sqrt(self.size - 2))
        if self.n * self.n != self.size - 2:
            print("O número de processos escravos (size - 2) deve ser um quadrado perfeito.")
            self.comm.Abort()

    def divide_image(self, frame):
        height, width, _ = frame.shape
        h_step = height // self.n
        w_step = width // self.n

        quadrants = []
        positions = []
        for i in range(self.n):
            for j in range(self.n):
                y_start = i * h_step
                y_end = (i + 1) * h_step if i != self.n - 1 else height
                x_start = j * w_step
                x_end = (j + 1) * w_step if j != self.n - 1 else width
                quadrant_data = frame[y_start:y_end, x_start:x_end]
                quadrant_name = f"quadrant_{i}_{j}"
                quadrants.append((quadrant_name, quadrant_data))
                positions.append((x_start, y_start))  # Armazena a posição inicial do quadrante
        return quadrants, positions, h_step, w_step

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

                # Divide a imagem em quadrantes e obtém suas posições
                quadrants, positions, h_step, w_step = self.divide_image(frame)

                # Envia os quadrantes para os processos escravos
                for idx, ((quadrant_name, quadrant_data), (x_start, y_start)) in enumerate(zip(quadrants, positions)):
                    dest_rank = idx + 2  # Processos escravos começam no rank 2
                    self.comm.send((quadrant_name, quadrant_data), dest=dest_rank, tag=11)

                # Recebe os resultados dos escravos
                results = {}
                for i in range(2, self.size):
                    # Recebe os dados serializados e desserializa
                    serialized_data = self.comm.recv(source=i, tag=22)
                    quadrant_name, result = pickle.loads(serialized_data)
                    results[quadrant_name] = result

                # Desenhar linhas para visualizar os quadrantes
                height, width, _ = frame.shape
                # Desenhar linhas horizontais
                for i in range(1, self.n):
                    y = i * h_step
                    cv2.line(frame, (0, y), (width, y), (0, 255, 0), 1)
                # Desenhar linhas verticais
                for j in range(1, self.n):
                    x = j * w_step
                    cv2.line(frame, (x, 0), (x, height), (0, 255, 0), 1)

                # Desenhar os elipses dos grãos de café detectados
                for idx, ((quadrant_name, _), (x_offset, y_offset)) in enumerate(zip(quadrants, positions)):
                    if quadrant_name in results:
                        result = results[quadrant_name]
                        ellipses = result.get('ellipses', [])

                        for ellipse in ellipses:
                            # Ajustar as coordenadas para o frame completo
                            center = (
                                int(ellipse['center'][0] + x_offset),
                                int(ellipse['center'][1] + y_offset)
                            )
                            axes = (
                                int(ellipse['axes'][0] / 2),
                                int(ellipse['axes'][1] / 2)
                            )
                            angle = ellipse['angle']

                            # Desenhar elipse no frame completo
                            cv2.ellipse(frame, center, axes, angle, 0, 360, (255, 0, 0), 2)

                # Exibe o frame
                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Enviar sinal de parada para os escravos
                    for i in range(2, self.size):
                        self.comm.send(None, dest=i, tag=99)
                    break

        except KeyboardInterrupt:
            print("Interrupção pelo usuário.")
            # Enviar sinal de parada para os escravos
            for i in range(2, self.size):
                self.comm.send(None, dest=i, tag=99)
            self.comm.Abort()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
                
    @override
    def work(self):
        self.webcam()
        pass

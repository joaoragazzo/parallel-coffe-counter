from mpi4py import MPI
import sys, cv2
import numpy as np
from roles.master import Master
from roles.slave import Slave
from roles.classifier import Classifier


def get_process_role(rank, comm):
    if rank == 0:
        return Master(comm)
    
    if rank == 1:
        return Classifier(comm)
    
    return Slave(comm)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

process = get_process_role(rank, comm)
process.work()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    sys.exit()

cv2.namedWindow('Superior Esquerdo', cv2.WINDOW_NORMAL)
cv2.namedWindow('Superior Direito', cv2.WINDOW_NORMAL)
cv2.namedWindow('Inferior Esquerdo', cv2.WINDOW_NORMAL)
cv2.namedWindow('Inferior Direito', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar o quadro.")
        break

    height, width = frame.shape[:2]

    mid_height = height // 2
    mid_width = width // 2

    top_left = frame[0:mid_height, 0:mid_width]
    top_right = frame[0:mid_height, mid_width:width]
    bottom_left = frame[mid_height:height, 0:mid_width]
    bottom_right = frame[mid_height:height, mid_width:width]

    region_size = 50

    q_height, q_width = top_left.shape[:2]
    center_y = q_height // 2
    center_x = q_width // 2

    y_start = max(center_y - region_size // 2, 0)
    y_end = min(center_y + region_size // 2, q_height)
    x_start = max(center_x - region_size // 2, 0)
    x_end = min(center_x + region_size // 2, q_width)

    center_region = top_left[y_start:y_end, x_start:x_end]

    mean_color = cv2.mean(center_region)[:3]  

    threshold = 53
    if all(c < threshold for c in mean_color):
        cv2.rectangle(top_left, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.putText(top_left, '!', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.rectangle(top_left, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.imshow('Superior Esquerdo', top_left)
    cv2.imshow('Superior Direito', top_right)
    cv2.imshow('Inferior Esquerdo', bottom_left)
    cv2.imshow('Inferior Direito', bottom_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

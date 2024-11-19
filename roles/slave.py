from mpi4py import MPI
from roles.worker import Worker
from typing import override
import cv2
import numpy as np
import pickle  

class Slave(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm

    def detect_coffee_beans(self, quadrant_data):
        # Step 1: Convert to HSV color space
        hsv = cv2.cvtColor(quadrant_data, cv2.COLOR_BGR2HSV)

        # Step 2: Define color ranges for coffee beans
        # Adjust these ranges based on experimentation
        # Dark roasted beans
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([230, 230, 100])

        # Light roasted beans
        lower_light = np.array([10, 50, 50])
        upper_light = np.array([30, 255, 255])

        # Green coffee beans
        lower_green = np.array([35, 25, 25])
        upper_green = np.array([85, 255, 255])

        # Combine masks
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        mask_light = cv2.inRange(hsv, lower_light, upper_light)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        combined_mask = cv2.bitwise_or(mask_dark, mask_light)
        combined_mask = cv2.bitwise_or(combined_mask, mask_green)

        # Step 3: Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # Step 4: Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Filter contours based on shape and size
        coffee_beans = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Minimum area threshold
                if len(cnt) >= 5:  # Required for fitEllipse
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (MA, ma), angle = ellipse
                    aspect_ratio = ma / MA if MA != 0 else 0
                    if 0.5 < aspect_ratio < 1.5:
                        coffee_beans.append({
                            'center': (x, y),
                            'axes': (MA, ma),
                            'angle': angle
                        })

        # Prepare the result
        result = {
            'count': len(coffee_beans),
            'ellipses': coffee_beans
        }

        return result

    @override
    def work(self):
        while True:
            status = MPI.Status()
            data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.tag == 99:
                # Sinal de parada
                break
            else:
                quadrant_name, quadrant_data = data

                # Processa o quadrante
                result = self.detect_coffee_beans(quadrant_data)

                # Serializa os dados antes de enviar
                serialized_data = pickle.dumps((quadrant_name, result))
                self.comm.send(serialized_data, dest=0, tag=22)

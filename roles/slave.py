from mpi4py import MPI
from roles.worker import Worker
from typing import override
import cv2
import numpy as np
import pickle  

class Slave(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm
        self.area_threshold_X = 0

    def detect_coffee_beans(self, quadrant_data, quadrant_position):
        row, col = quadrant_position

        ignore_borders = set()
        if row != 0:
            ignore_borders.add('top')
        if col != 0:
            ignore_borders.add('left')

        area_threshold_X = self.area_threshold_X  # Adjust this value as needed

        gray = cv2.cvtColor(quadrant_data, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # You can also try cv2.ADAPTIVE_THRESH_MEAN_C
            cv2.THRESH_BINARY_INV,
            15,  # Block size (must be odd)
            2    # Constant subtracted from the mean or weighted mean
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_bg = cleaned
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        quadrant_data_color = cv2.cvtColor(quadrant_data, cv2.COLOR_BGR2RGB)  # Convert to RGB for watershed
        markers = cv2.watershed(quadrant_data_color, markers)
        bean_images = []
        coffee_beans = []
        for marker_id in range(2, np.max(markers) + 1):  # Skip background and boundary markers
            mask = np.uint8(markers == marker_id)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            if area < 100 or area > 5000:  # Adjust area thresholds as needed
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            touches_ignored_border = False
            if 'left' in ignore_borders and x <= 0:
                touches_ignored_border = True
            if 'top' in ignore_borders and y <= 0:
                touches_ignored_border = True
            if touches_ignored_border and area <= area_threshold_X:
                continue

            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 1.5:
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    coffee_beans.append({
                        'center': ellipse[0],
                        'axes': ellipse[1],
                        'angle': ellipse[2]
                    })
                else:
                    center = (x + w / 2, y + h / 2)
                    axes = (w / 2, h / 2)
                    angle = 0
                    coffee_beans.append({
                        'center': center,
                        'axes': (w, h),
                        'angle': angle
                    })

                x_roi = max(x - 5, 0)
                y_roi = max(y - 5, 0)
                w_roi = min(w + 10, quadrant_data.shape[1] - x_roi)
                h_roi = min(h + 10, quadrant_data.shape[0] - y_roi)
                bean_img = quadrant_data[int(y_roi):int(y_roi + h_roi), int(x_roi):int(x_roi + w_roi)]
                bean_images.append(bean_img)

        result = {
            'count': len(coffee_beans),
            'ellipses': coffee_beans,
            'bean_images': bean_images
        }

        
        if self.comm.Get_rank() == 3:
            cv2.imshow('Gray (Rank 3)', gray)
            cv2.imshow('Blurred (Rank 3)', blurred)
            cv2.imshow('Thresholded (Rank 3)', thresh)
            cv2.imshow('Cleaned (Rank 3)', cleaned)
            cv2.imshow('Distance Transform (Rank 3)', dist_transform / dist_transform.max())
            cv2.waitKey(1)  # Allow OpenCV to process GUI events

        return result

    @override
    def work(self):
        try:
            while True:
                status = MPI.Status()
                data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                if status.tag == 99:
                    break
                else:
                    quadrant_name, quadrant_data, parameters_att = data

                    for param_name, param_value in parameters_att.items():
                        setattr(self, param_name, param_value)

                    result = self.detect_coffee_beans(quadrant_data, quadrant_name)

                    serialized_data = pickle.dumps((quadrant_name, result))
                    self.comm.send(serialized_data, dest=0, tag=22)
        finally:
            if self.comm.Get_rank() == 3:
                cv2.destroyAllWindows()

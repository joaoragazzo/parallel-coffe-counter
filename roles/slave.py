from mpi4py import MPI
from roles.worker import Worker
from typing import override
import cv2
import numpy as np
import pickle
from collections import Counter

class Slave(Worker):
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm
        self.area_threshold_X = 0
        self.minThreshold = 30
        self.maxThreshold = 255
        self.firstMorphElipse = 8
        self.secondMorphElipse = 18
        self.firstErodeIterations = 1
        self.secondErodeIterations = 1
        self.minDistance = 10
        self.show_cams = 0

        self.manual_classification_enabled = False

    def detect_coffee_beans(self, quadrant_data, quadrant_position):
        row, col = quadrant_position

        ignore_borders = set()
        if row != 0:
            ignore_borders.add('top')
        if col != 0:
            ignore_borders.add('left')

        area_threshold_X = 100  

        gray = cv2.cvtColor(quadrant_data, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, self.minThreshold, self.maxThreshold, cv2.THRESH_BINARY_INV)

        segments_small_kernel = []

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.firstMorphElipse, self.firstMorphElipse))
        eroded_small = cv2.erode(thresh, kernel_small, iterations=self.firstErodeIterations)

        num_labels_small, labels_im_small = cv2.connectedComponents(eroded_small)

        for label in range(1, num_labels_small):
            mask = labels_im_small == label
            area = np.sum(mask)
            if area < 300:  
                segments_small_kernel.append(mask)

        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.secondMorphElipse, self.secondMorphElipse))
        eroded_large = cv2.erode(thresh, kernel_large, iterations=self.secondErodeIterations)

        num_labels_large, labels_im_large = cv2.connectedComponents(eroded_large)

        segments_all = []
        for label in range(1, num_labels_large):
            mask = labels_im_large == label
            segments_all.append(mask)

        segments_combined = segments_small_kernel + segments_all

        contours = []
        for seg in segments_combined:
            seg = np.uint8(seg * 255)
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours.append(cnts[0])

        filtered_contours = []
        min_distance = 10  
        centers = []

        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            too_close = False
            for (c_x, c_y) in centers:
                distance = np.sqrt((cx - c_x) ** 2 + (cy - c_y) ** 2)
                if distance < min_distance:
                    too_close = True
                    break
            if not too_close:
                centers.append((cx, cy))
                filtered_contours.append(cnt)

        bean_images = []
        coffee_beans = []
        for cnt in filtered_contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 5000:  
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            touches_ignored_border = False
            if 'left' in ignore_borders and x <= 0:
                touches_ignored_border = True
            if 'top' in ignore_borders and y <= 0:
                touches_ignored_border = True

            if touches_ignored_border and area <= area_threshold_X:
                continue

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
            bean_img = quadrant_data[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
            bean_images.append(bean_img)

        if self.manual_classification_enabled:
            result = {
                'count': len(coffee_beans),
                'ellipses': coffee_beans,
                'bean_images': bean_images
            }
            
        else:
            self.comm.send(bean_images, dest=1, tag=11)
            types = self.comm.recv(source=1, tag=22)

            for idx, bean in enumerate(coffee_beans):
                bean['type'] = types[idx]

            type_counts = Counter(types)

            result = {
                'count': len(coffee_beans),
                'ellipses': coffee_beans, 
                'type_counts': type_counts
            }

        if self.show_cams:
            cv2.imshow(f'Thresholded (Rank {self.comm.Get_rank()})', thresh)
            cv2.imshow(f'Eroded Small Kernel (Rank {self.comm.Get_rank()})', eroded_small)
            cv2.imshow(f'Large Kernel (Rank {self.comm.Get_rank()})', eroded_large)
            cv2.waitKey(1)  
        else:
            cv2.destroyAllWindows()

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
                    quadrant_name, quadrant_data, parameters_att, self.show_cams, self.manual_classification_enabled = data

                    for param_name, param_value in parameters_att.items():
                        setattr(self, param_name, param_value)

                    result = self.detect_coffee_beans(quadrant_data, quadrant_name)

                    serialized_data = pickle.dumps((quadrant_name, result))
                    self.comm.send(serialized_data, dest=0, tag=22)
        finally:
            cv2.destroyAllWindows() 


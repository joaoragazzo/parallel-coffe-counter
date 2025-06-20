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
        self.model = tf.keras.models.load_model('./neural.keras')

    @override
    def work(self):
        while True:
            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == 99:  
                break
            else:
                bean_images = data
                types = []
                for bean_img in bean_images:
                    bean_img_resized = cv2.resize(bean_img, (224, 224))
                    bean_img_normalized = bean_img_resized / 255.0
                    bean_img_normalized = bean_img_normalized.reshape(1, 224, 224, 3)
                    predict = self.model.predict(bean_img_normalized)
                    predicted_class_idx = np.argmax(predict[0])
                    predicted_class = ['Black', 'Green', 'White', 'Roasted'][predicted_class_idx]
                    types.append(predicted_class)
                self.comm.send(types, dest=source, tag=22)

from mpi4py import MPI
from worker import Worker
from typing import override
import cv2
import tensorflow as tf
import numpy as np
import os


class Classifier(Worker):
    def __init__(self, comm: MPI.Comm, **kargs):
        self.comm = comm

    @override
    def work():
        pass


def read_image(path):
    images = []
    
    if not os.path.isdir(path):
        print(f"A pasta '{path}' não existe ou não é válida.")
        return images

    for file in os.listdir(path):
        if file.lower().endswith('.jpg'):
            path_image = os.path.join(path, file)
            try:
                image = cv2.imread(path_image)
                
                if image is None:
                    print(f"Erro ao carregar a imagem '{path_image}'.")
                    continue
                
                images.append(image)
            except Exception as e:
                print(f"Erro ao processar a imagem '{path_image}': {e}")
    
    return images

def ramdomize(train_d, train_g, train_l, train_m):
    np.random.seed(None)
        
    targets_d = [0] * len(train_d)
    targets_g = [1] * len(train_g)
    targets_l = [2] * len(train_l)
    targets_m = [3] * len(train_m)

    train_all = np.array(train_d + train_g, train_l, train_m)
    targets_all = np.array(targets_d + targets_g + targets_l + targets_m)

    indices = np.arange(len(train_all))
    np.random.shuffle(indices)

    train_all_shuffled = train_all[indices]
    targets_shuffled = targets_all[indices]

    return train_all_shuffled, targets_shuffled

def prepare_data():
    path_d = "..\\images\\train\\Dark"
    train_d = read_image(path_d)
    print(f"{len(train_d)} imagens foram carregadas.")

    path_g = "..\\images\\train\\Green"
    train_g = read_image(path_g)
    print(f"{len(train_g)} imagens foram carregadas.")

    path_l = "..\\images\\train\\Light"
    train_l = read_image(path_l)
    print(f"{len(train_l)} imagens foram carregadas.")

    path_m = "..\\images\\train\\Medium"
    train_m = read_image(path_m)
    print(f"{len(train_m)} imagens foram carregadas.")

    train_all, targets = ramdomize(train_d, train_g, train_l, train_m)

    targets = tf.keras.utils.to_categorical(targets, num_classes=4)

    return train_all, targets

def create_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),                            
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    return model


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
#model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

while True:
    command = input("Enter '1' to train the model, '2' to predict images of autistic individuals, '3' to predict images of non-autistic individuals, '4' to save the model, '5' to load model, or '0' to exit: ")

    if command == '1':
        print("Training model...")
        model.fit(train_all, targets, epochs=10, batch_size=8, validation_data=(valid_all, v_targets))
    
    elif command == '2':
        path_images = 'autism\\images\\valid\\autistic'
        for file in os.listdir(path_images):
            path_image = os.path.join(path_images, file)
            new_image = cv2.imread(path_image, cv2.IMREAD_COLOR)
            if new_image is None:
                print(f"Erro ao carregar a imagem: {path_image}")
                continue
            new_image = cv2.resize(new_image, (hw, hw)) / 255.0
            new_image = new_image.reshape(1, hw, hw, 3)

            predict = model.predict(new_image)
            print(f"Image: {file}, Predicted Probability: {predict[0][0]:.2f}, Classification: {'Autistic' if predict[0][0] > 0.5 else 'Non-Autistic'}")

    elif command == '3':
        path_images = 'autism\\images\\valid\\non_autistic'
        for file in os.listdir(path_images):
            path_image = os.path.join(path_images, file)
            new_image = cv2.imread(path_image, cv2.IMREAD_COLOR)
            if new_image is None:
                print(f"Erro ao carregar a imagem: {path_image}")
                continue
            new_image = cv2.resize(new_image, (hw, hw)) / 255.0
            new_image = new_image.reshape(1, hw, hw, 3)

            predict = model.predict(new_image)
            print(f"Image: {file}, Predicted Probability: {predict[0][0]:.2f}, Classification: {'Autistic' if predict[0][0] > 0.5 else 'Non-Autistic'}")

    elif command == '4':
        model.save('autism\\neural_networks\\modelo.keras')
    elif command == '5':
        try:
            model = tf.keras.models.load_model('autism\\neural_networks\\modelo_1.keras')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    elif command == '0':
        print("Exiting loop...")
        break
    else:
        print("Invalid command! Type '1', '2', '3', '4', '5', or '0'.")

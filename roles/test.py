import cv2
import tensorflow as tf
import numpy as np
import os

def read_image(path):
    images = []
    
    if not os.path.isdir(path):
        print(f"A pasta '{path}' não existe ou não é válida.")
        return images

    for file in os.listdir(path):
        if file.lower().endswith('.png'):
            path_image = os.path.join(path, file)
            try:
                image = cv2.imread(path_image)
                
                if image is None:
                    print(f"Erro ao carregar a imagem '{path_image}'.")
                    continue
                imagem_redimensionada = cv2.resize(image, (224, 224))
                
                imagem_normalizada = imagem_redimensionada / 255.0
                
                images.append(imagem_normalizada)
            except Exception as e:
                print(f"Erro ao processar a imagem '{path_image}': {e}")
    
    return images

def ramdomize(train_d, train_g, train_l, train_m):
    np.random.seed(None)
        
    targets_d = [0] * len(train_d)
    targets_g = [1] * len(train_g)
    targets_l = [2] * len(train_l)
    targets_m = [3] * len(train_m)

    train_all = np.array(train_d + train_g + train_l + train_m)
    targets_all = np.array(targets_d + targets_g + targets_l + targets_m)

    indices = np.arange(len(train_all))
    np.random.shuffle(indices)

    train_all_shuffled = train_all[indices]
    targets_shuffled = targets_all[indices]

    return train_all_shuffled, targets_shuffled

def prepare_data():
    path_d = ".\\images\\train\\Dark"
    train_d = read_image(path_d)
    print(f"{len(train_d)} imagens foram carregadas.")

    path_g = ".\\images\\train\\Green"
    train_g = read_image(path_g)
    print(f"{len(train_g)} imagens foram carregadas.")

    path_l = ".\\images\\train\\Light"
    train_l = read_image(path_l)
    print(f"{len(train_l)} imagens foram carregadas.")

    path_m = ".\\images\\train\\Medium"
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

train, target = prepare_data()
model = create_model()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

while True:
    command = input(
        "Enter '1' to train the model, '2' to validate images in 'dark', '3' to validate images in 'green', '4' to validate images in 'light', '5' to validate images in 'medium', '6' to save the model, '7' to load the model, or '0' to exit: "
    )

    if command == '1':
        print("Training model...")
        model.fit(train, target, epochs=10, batch_size=8, validation_split=0.3)

    elif command in ['2', '3', '4', '5']:
        # Mapear os diretórios com base no comando
        paths = {
            '2': '.\\images\\test\\Dark',
            '3': '.\\images\\test\\Green',
            '4': '.\\images\\test\\Light',
            '5': '.\\images\\test\\Medium',
        }
        selected_path = paths.get(command, None)

        if selected_path:
            print(f"Validating images in '{selected_path}'...")
            for file in os.listdir(selected_path):
                path_image = os.path.join(selected_path, file)
                new_image = cv2.imread(path_image, cv2.IMREAD_COLOR)
                if new_image is None:
                    print(f"Error loading image: {path_image}")
                    continue
               
                new_image = cv2.resize(new_image, (224, 224)) / 255.0
                new_image = new_image.reshape(1, 224, 224, 3)
                
                predict = model.predict(new_image)
                predicted_class_idx = np.argmax(predict[0])  # Índice da classe com maior probabilidade
                predicted_class = ['Dark', 'Green', 'Light', 'Medium'][predicted_class_idx]  # Mapear índice para rótulos
                print(
                    f"Image: {file}, Predicted Class: {predicted_class}, Probabilities: {predict[0]}"
                )

    elif command == '6':
        print("Saving the model...")
        model.save('.\\neural_activation\\bean_classification_model.keras')
        print("Model saved successfully.")

    elif command == '7':
        print("Loading the model...")
        try:
            model = tf.keras.models.load_model('.\\neural_activation\\bean_classification_model_final.keras')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    elif command == '8':
        paths = '.\\images\\test\\Extra'

        print(f"Validating images in '{paths}'...")
        for file in os.listdir(paths):
            path_image = os.path.join(paths, file)
            new_image = cv2.imread(path_image, cv2.IMREAD_COLOR)
            if new_image is None:
                print(f"Error loading image: {path_image}")
                continue
            
            new_image = cv2.resize(new_image, (224, 224)) / 255.0
            new_image = new_image.reshape(1, 224, 224, 3)
            
            predict = model.predict(new_image)
            predicted_class_idx = np.argmax(predict[0])  # Índice da classe com maior probabilidade
            predicted_class = ['Dark', 'Green', 'Light', 'Medium'][predicted_class_idx]  # Mapear índice para rótulos
            print(
                f"Image: {file}, Predicted Class: {predicted_class}, Probabilities: {predict[0]}"
            )
    elif command == '0':
        print("Exiting loop...")
        break

    else:
        print("Invalid command! Type '1', '2', '3', '4', '5', '6', '7', or '0'.")

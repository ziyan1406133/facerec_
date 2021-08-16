from tensorflow import keras
from PIL import Image

import os
import numpy as np
import zipfile
import cv2
import imutils

class Net():
    def __init__(self):
        self.getDataset()

        self.model = keras.Sequential([
            keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(100,100,1)),
            # keras.layers.InputLayer(input_shape=(10000,)),
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.num_classes)
        ])
        # self.model = keras.Sequential([
        #     keras.layers.InputLayer(input_shape=(100, 100, 3)),
        #     keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(10)
        # ])
        
        print("\nTraining Model, Please Wait ...\n")
        self.compile()
        self.train()
        

    def compile(self):
        self.model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    def train(self):
        self.history = self.model.fit(
                self.train_images,
                self.train_classes,
                epochs=15,
                validation_split=0.1,
                verbose=True
        )

    def predict(self, x):
        prediction = self.model.predict(
            x, verbose=False
        )
        
        return np.argmax(prediction, axis=1)[0]

    def getDataset(self):
        path = "cropped_dataset"
        detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
        self.train_images = list()
        self.target_classes = list()
        self.train_classes = list()
        folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))] 
    
        for i in range(len(folders)):
            self.target_classes.append(folders[i])
            gambar = [f for f in os.listdir(os.path.join(path, folders[i]))]    
            for g in gambar:
                filepath = os.path.join(os.path.join(path, folders[i]), g)
                img = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)

                img_numpy = keras.preprocessing.image.img_to_array(img)
                # img_numpy = np.array(img, 'uint8')
                self.train_images.append(img_numpy)
                self.train_classes.append(i)

        self.num_classes = len(folders)
        self.train_images = np.array(self.train_images)
        self.target_classes = np.array(self.target_classes)
        self.train_classes = np.array(self.train_classes)
        print(self.target_classes)
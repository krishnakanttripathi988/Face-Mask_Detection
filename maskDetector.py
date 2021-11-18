
#############################################################################################
'''<<WORKING WITH IMPORTS>>
'''
from numpy import testing
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
# import argparse
import os
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
"""defining path to save model and path to access raw dataset"""
#######################################################################
model_path = "mask.model"
dataset_raw = "dataset"
#######################################################################





# class CustomCallback(keras.callbacks.Callback):

#     def on_epoch_end(self, epoch, logs=None):
#         print("this iteration is: ", epoch)


def mask_trainer():
    dataset = list(paths.list_images(dataset_raw))
    labels = []
    values = []
    for data in dataset:
        label = data.split(os.path.sep)[-2]
        childs = load_img(data, target_size=(224, 224))
        childs = img_to_array(childs)
        childs = preprocess_input(childs)
        values.append(childs)
        labels.append(label)
    values = np.array(values, np.float32)
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    (trainX, testX, trainY, testY) = train_test_split(
        values, labels, test_size=0.20, stratify=labels, random_state=42)
    grid_view = ImageDataGenerator(rotation_range=50,	zoom_range=0.20, width_shift_range=0.224,height_shift_range=0.224, shear_range=0.20, horizontal_flip=True, vertical_flip=True)
    facenet = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), input_tensor=Input(shape=(224, 224, 3)))
    dum_model = facenet.output
    dum_model = AveragePooling2D(pool_size=(7, 7))(dum_model)
    dum_model = Flatten(name="flatten")(dum_model)
    dum_model = Dense(128, activation="relu")(dum_model)
    dum_model = Dropout(0.5)(dum_model)
    dum_model = Dense(2, activation="softmax")(dum_model)
    model = Model(inputs=facenet.input, outputs=dum_model)
    for i in facenet.layers:
        i.trainable = False
    alpha = 3e-4
    iters = 20
    BS = 32
    opt = Adam(lr=alpha, decay=alpha / iters)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    final = model.fit(grid_view.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY), validation_steps=len(testX) // BS, epochs=iters)
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testY.argmax(axis=1),predIdxs,	target_names=lb.classes_))
    model.save(model_path, save_format="h5")



def mask_checker():
    def detect(frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:

            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)


    prototxtPath = "dataset/face_detector/deploy.prototxt"
    weightsPath = "./dataset/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model("mask.model")
    
    vs = VideoStream(src=0).start()
    time.sleep(3.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        (locs, preds) = detect(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            if mask < withoutMask:
                path = os.path.abspath("Alarm.wav")
                playsound(path)

        cv2.imshow("Mask Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


    cv2.destroyAllWindows()
    vs.stop()






if __name__ == "__main__":
    a = input("Enter Train to Train the model or enter Detect to Detect the face: ")
    if(a == "Train"):
        mask_trainer()
    elif(a == "Detect"):
        mask_checker()
    else:
        print("Please enter a valid option")


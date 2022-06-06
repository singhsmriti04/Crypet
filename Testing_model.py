import cv2
import numpy as np
from os import listdir                      #Used to fetch data from a directory
from os.path import isfile, join
from tkinter import *

def test():
    data_path = 'C:/Users/LENOVO/Desktop/cv faces/'  # location where our data is stored
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    training_data, labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]                                        #To get the path of each file(image)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                        #convert each image to grayscale image
        training_data.append(np.asarray(images, dtype=np.uint8))                     #make a list of all training images
        labels.append(i)                                                             #list of all indices

    labels = np.asarray(labels, dtype=np.int32)                                      #convert label(list) to array

    model = cv2.face.LBPHFaceRecognizer_create()                                     #LocalBinaryPatternHistogram-

    model.train(np.asarray(training_data), np.asarray(labels))

    # print("Model training completed.")

    face_classifier = cv2.CascadeClassifier(
        'C:/Users/LENOVO/PycharmProjects/face_recog/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return img, []

        # to draw rect over the face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255),
                          2)  # thickness of rect= 2, color of rect -(0,255,255)
            roi = img[y:y + h, x:x + w]  # region of interest= roi
            roi = cv2.resize(roi, (200, 200))

        return img, roi

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) + '% confidence it is user'

            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255))

            if confidence > 85:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)

                # window = Tk()
                # add widgets here

                # window.title('Hello Python')
                # window.geometry("300x200+10+20")
                # window.mainloop()


            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

            pass

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

test()




import cv2
import numpy as np

def train():
    face_classifier = cv2.CascadeClassifier(
        'C:/Users/LENOVO/PycharmProjects/face_recog/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    # To extract the features of the face

    def face_extractor(img):
        gray = cv2.cvtColor(img,
                            cv2.COLOR_BGR2GRAY)  # Since it is easy to work with gray scale images as compared to BGR images
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # We have taken scaling factor = 1.3  Min neighbours = 5

        if faces is ():  # If faces has no value we will return NONE
            return None

        for (x, y, w, h) in faces:  # if faces has some value than we will crop the face coordinates , width , height
            cropped_face = img[y:y + h, x:x + w]  # y is row value, x is column value

        return cropped_face

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()

        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame),
                              (200, 200))  # The frame of the camera should be of the size of our face.
            face = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)

            file_name_path = 'C:/Users/LENOVO/Desktop/cv faces/user' + str(
                count) + '.jpg'  # Place where the image will be stored
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass
        if cv2.waitKey(1) == 13 or count == 50:
            break

    cap.release()
    cv2.destroyAllWindows()

train()

print('Collecting samples completed!')











from tkinter import *
from tkinter import messagebox
import cv2
import numpy as np
from os import listdir                                           # Used to fetch data from a directory
from os.path import isfile, join
import webbrowser
import mysql.connector


#db_connection = mysql.connector.connect(user='Shivani', password='SHIVANi1999@')
#db_cursor = db_connection.cursor(buffered = True)
#db_cursor.execute("show databases")

#for i in db_cursor:
#    print(i)
 

def test():
    data_path = 'C:/Users/LENOVO/Desktop/cv faces/'  # location where our data is stored
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    training_data, labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]  # To get the path of each file(image)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # convert each image to grayscale image
        training_data.append(np.asarray(images, dtype=np.uint8))  # make a list of all training images
        labels.append(i)  # list of all indices

    labels = np.asarray(labels, dtype=np.int32)  # convert label(list) to array

    model = cv2.face.LBPHFaceRecognizer_create()  # LocalBinaryPatternHistogram-

    model.train(np.asarray(training_data), np.asarray(labels))

    print("Model training completed.")

    face_classifier = cv2.CascadeClassifier(
        'C:/Users/LENOVO/PycharmProjects/face_recog/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return img, []

        # to draw rectangle over the face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255),
                          2)  # thickness of rect= 2, color of rect -(0,255,255)
            roi = img[y:y + h, x:x + w]  # region of interest= roi
            roi = cv2.resize(roi, (200, 200))

        return img, roi

    cap = cv2.VideoCapture(0)
    win_name = 'Face Authetication'
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

            if confidence > 75:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
                cv2.waitKey(500)
                cap.release()
                cv2.destroyAllWindows()

                global window

                window = Tk()
                # add widgets here

                window.title('Congrats, You are logged in now!')
                window.geometry("300x250")

                btn = Button(window, text="Click here to move to transaction page", height="5", width="32",
                             command=OpenUrl)
                btn.pack(pady=60)

                # Button(text="Click Me", height="4", width="40").pack()

                window.mainloop()

                cap.release()
                cv2.destroyAllWindows()

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

def train():
    face_classifier = cv2.CascadeClassifier(
        'C:/Users/LENOVO/PycharmProjects/face_recog/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    # To extract the features of the face

    def face_extractor(img, size=0.5):
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
        if cv2.waitKey(1) == 13 or count == 100:           
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Collecting samples completed!')
    print('Congratulation! You have been registered now.')


def OpenUrl():
    window.destroy()
    url = 'http://localhost:3000/'
    webbrowser.open_new(url)

def login():                                                          #Function for user login
    screen.destroy()
    test()

def register():                                                       #Function for registering new user
    screen.destroy()
    train()



screen = Tk()                                                          #Main screen
screen.geometry("600x500")
screen.title('Face Authentication')

def log():
    global userID
    print("Enter your email and password")
    email = user.get()
    Password = pas.get()
    cur = connection.cursor()
    query = "SELECT email , password FROM users"
    cur.execute(query)
    for (e, P) in cur:
        if email == e and Password == P:
            login = True
            break
        else:
            login = False
    userID = email
    if login == True:
        print("Logged in successfully as", userID)
    elif login == False:
        print("Wrong Entry")

    cur.close()
    connection.close()



def LoginPage():

    screen.destroy()
    login_screen = Tk()
    login_screen.title("Login")
    login_screen.geometry("300x250")
    Label(login_screen, text="Please enter login details").pack()
    Label(login_screen, text="").pack()
    username = Label(login_screen, text="Username").pack()
    user = Entry(login_screen, textvariable="username")
    user.pack()
    Label(login_screen, text="").pack()
    password = Label(login_screen, text="Password").pack()
    pas = Entry(login_screen, textvariable="password", show= '*')
    pas.pack()
    Label(login_screen, text="").pack()
    submit = Button(login_screen, text="Login", width=10, height=1, command=log).pack()
    login_screen.mainloop()

Label(text="Face Authentication", background="grey", width="800", height="3", font=('Calibri', 13)).pack()
Label(text="").pack()
Button(text="Login Using Face", height="5", width="50", command=login).pack()
Label(text="").pack()
Button(text="Log In Using Pin", height="5", width="50", command=LoginPage).pack()
Label(text="").pack()
Button(text="Register", height="5", width="50", command=register).pack()


screen.mainloop()





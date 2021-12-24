import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import threading


emotional_model = Sequential()
emotional_model.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=[48, 48, 1]))
emotional_model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=2, strides=2))
emotional_model.add(Dropout(0.25))
emotional_model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=2, strides=2))
emotional_model.add(Dropout(0.25))
emotional_model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=2, strides=2))
emotional_model.add(Dropout(0.25))
emotional_model.add(Flatten())
emotional_model.add(Dense(units=1024, activation='relu'))
emotional_model.add(Dropout(0.5))
emotional_model.add(Dense(7, activation = 'softmax'))

emotional_model.load_weights('emojify.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: '    Angry   ', 1: '    Disgusted   ', 2: '  Fearful   ', 3:'    Happy   ', 4:'      Neutral     ', 5:'  Sad   ', 6:'    Surprised   '}
cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dist = {0:cur_path+'/emojis/angry.png', 1:cur_path+'/emojis/disgusted.png', 2:cur_path+'/emojis/fearful.png', 3:cur_path+'/emojis/happy.png', 4:cur_path+'/emojis/neutral.png', 5:cur_path+'/emojis/sad.png', 6:cur_path+'/emojis/surprised.png'}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype = np.uint8)
global cap1
show_text = [0]
global frame_number

def show_subject():
    cap1 = cv2.VideoCapture("C:/Users/user/Pictures/Camera Roll/WIN_20211221_13_15_31_Pro.mp4")
    if not cap1.isOpened():
        print("Can't open the camera")
    global frame_number
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1
    if frame_number >= length:
        exit()
    cap1.set(1, frame_number)
    
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600, 500))
    bounding_box = cv2.CascadeClassifier("C:/Users/user/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor = 1.3, minNeighbors = 5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotional_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if flag1 is None:
        print('Major error!')
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image = img)
        lmain.imgtk = imgtk
        lmain.configure(image = imgtk)

        root.update()
        root.after(10, show_subject)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_avatar():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image = img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text = emotion_dict[show_text[0]], font = ('arial', 45, 'bold'))
    lmain2.configure(image = imgtk2)
    root.update()
    root.after(10, show_avatar)

if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()
    lmain = tk.Label(master = root, padx =50, bd = 10)
    lmain2 = tk.Label(master = root, bd = 10)
    lmain3 = tk.Label(master = root, bd = 10, fg = '#CDCDCD', bg = 'black')
    lmain.pack(side = LEFT)
    lmain.place(x = 50, y = 250)
    lmain3.pack()
    lmain3.place(x = 860, y = 150)
    lmain2.pack(side = RIGHT)
    lmain2.place(x = 800, y = 250)

    root.title('Photo to Emoji')
    root.geometry('1400x800+100+10')
    root['bg'] = 'black'
    exitButton = Button(root, text = 'Quit', fg = 'red', command = root.destroy, font = ('arial', 25, 'bold')).pack(side = BOTTOM)
    t1 = threading.Thread(target = show_subject)
    t2 = threading.Thread(target = show_avatar, daemon = True)
    t1.start()
    t2.start()
    #t1.join()
    #t2.join()
    #show_subject()
    #show_avatar()
    root.mainloop()

#Video Files:
#"C:\Users\user\Pictures\Camera Roll\WIN_20211219_13_41_27_Pro.mp4"
#"C:/Users/user/Pictures/Camera Roll/WIN_20211221_11_40_54_Pro.mp4"
#"C:/Users/user/Pictures/Camera Roll/WIN_20211221_13_12_37_Pro.mp4"
#"C:/Users/user/Pictures/Camera Roll/WIN_20211221_13_15_31_Pro.mp4"
#"C:/Users/user/Pictures/Camera Roll/WIN_20211223_13_32_23_Pro.mp4"
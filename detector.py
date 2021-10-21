import numpy as np
import cv2
import os

def creatDir(name, path=''):

    if not os.path.exists(f'{path}/{name}'):

        os.makedirs(f'{path}/{name}')

def saveFace():

    global saveface
    global lastName

    saveface = True

    creatDir('train')

    print("Qual o seu nome?")

    name = input()

    lastName = name

    creatDir(name,'train')


def saveImg(img):

    global lastName

    qtd = os.listdir(f'train/{lastName}')

    cv2.imwrite(f'train/{lastName}/{str(len(qtd))}.jpg', img)

def trainData():

    global recognizer
    global trained
    global persons

    trained = True

    persons = os.listdir('train')

    ids = []

    faces = []

    for i,p in enumerate(persons):

        i += 1

        for f in os.listdir(f'train/{p}'):

            img = cv2.imread(f'train/{p}/{f}',0)

            faces.append(img)

            ids.append(i)

    recognizer.train(faces, np.array(ids))


lastName = ''


saveface = False
savefaceC = 0


trained = False
persons = []


cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()



while(True):
    

    _, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for i, (x,y,w,h) in enumerate(faces):


        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


        roi_gray = gray[y:y+h, x:x+w]


        resize = cv2.resize(roi_gray, (50, 50)) 


        if trained:

            idf, conf = recognizer.predict(resize)


            nameP = persons[idf-1]

            if conf < 100:
                cv2.putText(frame,nameP,(x+5,y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)

            else:
                cv2.putText(frame,nameP,(x+5,y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)


            cv2.putText(frame,'TREINADO',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
        else:
            cv2.putText(frame,'NAO TREINADO',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
            

        if saveface:

            cv2.putText(frame,str(savefaceC),(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)

            savefaceC += 1

            saveImg(resize)

        if savefaceC > 50:

            savefaceC = 0

            saveface = False



    cv2.putText(frame,'Espaco para salvar face',(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)


    cv2.imshow('Reconhecimento Facial',frame)


    key = cv2.waitKey(15)


    if key == 116:
        trainData()


    if key == 32:
        saveFace()


    if key & 0xFF == ord('q'):
        break
        

cap.release()

cv2.destroyAllWindows()
#Yüz tanıma programı
import os
import colorama
from colorama import Fore, Back, Style
from cv2.cv2 import CascadeClassifier

colorama.init()
import time
import cv2
import numpy as np
from PIL import Image
import os,json
time.sleep(1)
"""



"""

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
global user
user = input("Kullanıcı adı giriniz : ")
#face_id = input('\n Admin id girin ==>  ')
print("\n {BİLGİ] Kamereya bakın ve bekleyin  ...")
say = 0
os.mkdir('dataset/'+user)


while(True):
    ret, cerceve = cam.read()
    cerceve = cv2.flip(cerceve, 1)
    gri = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gri, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(cerceve, (x,y), (x+w,y+h), (255,0,0), 2)
        say += 1
        path = "dataset/"+user+"/"
        cv2.imwrite(path+str(say) + ".jpg", gri[y:y +h, x:x +w])
        cv2.imshow('DATA', cerceve)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        print("HATA KODU 785")
        time.sleep(3)
        break
    elif say >= 550:
        print("HATA KODU 145")
        time.sleep(3)
        break


cam.release()
cv2.destroyAllWindows()

#Day 2 - 30.07.2020 Bu projenin hayatının en uzun projesi olcağını biliyorsun.
#Sabırlı ol.


yol = 'dataset'
tani = cv2.face.LBPHFaceRecognizer_create()
detector = (cv2.CascadeClassifer(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"));

def getImagesAndLabels(yol):
    #imagePaths = [os.path.join(yol,f) for f in os.listdir(yol)]
    faceSamples=[]
    ids = []
    labels = []
    klasorler = os.listdir(yol)
    dictionary = {}

    for i,kl in enumerate(klasorler):
        dictionary[kl]=int(i)

    f = open("ids.json", "w")
    a = json.dump(dictionary,f)
    f.close()

    for kl in klasorler:
        for res in os.listdir(os.path.join(yol,kl)):
            PIL_img = Image.open(os.path.join(yol,kl,res)).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(dictionary[kl])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

    return faceSamples, ids

faces,ids = getImagesAndLabels(yol)


tani.train(faces, np.array(ids))
tani.write('trainer.yml')

tani = cv2.face.LBPHFaceRecognizer_create()
tani.read('trainer.yml')
cascadePath = (cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0


dictionary={}
names = []
dosya = open("ids.json","r")
dictionary = json.load(dosya)
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

for key,value in dictionary.items():
    names.append(key)


while True:
    ret, cerceve = cam.read()
    cerceve = cv2.flip(cerceve, 1)
    gri = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gri,scaleFactor=1.5,minNeigbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(cerceve, (x, y), (x + w, y + h), (0, 255 , 0), 2)
        id, oran = tani.predict(gri[y:y + h, x:x + w])
        print(id)

        if (oran < 70):
            id = names[id]
        else:
            id = "Bilinmiyor"

        cv2.putText(cerceve, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    cv2.imshow('KAMERA', cerceve)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        print("HATA KODU 522")
        time.sleep(4)
        break


cam.release()
cv2.destroyAllWindows()
print("TAMAMLANDI")
time.sleep(3)
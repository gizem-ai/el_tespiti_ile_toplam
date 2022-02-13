import cv2 as cv
import numpy as np
import os
import cv2
import random

kamera = cv.VideoCapture(0)
kernel = np.ones((12,12),np.uint0)


def ResimFarkiBul (resim1,resim2):
    resim2 = cv.resize(resim2,(resim1.shape[1],resim1.shape[0]))
    farkResim = cv.absdiff(resim1,resim2)
    farkSayi = cv.countNonZero(farkResim)
    return farkSayi

def veriYukleme():
    veriIsimleri = []
    veriResimler = []

    dosyalar = os.listdir("tanimlanan/")
    for dosya in dosyalar:
        veriIsimleri.append(dosya.replace(".jpg",""))
        veriResimler.append(cv.imread("tanimlanan/"+dosya,0))
    return veriIsimleri, veriResimler

def siniflandir(resim,veriIsimleri,veriResimler):
    minIndex = 0
    minDeger = ResimFarkiBul(resim,veriResimler[0])
    for t in range(len(veriIsimleri)):
        farkDeger = ResimFarkiBul(resim,veriResimler[t])
        if(farkDeger<minDeger):
            minDeger = farkDeger
            minIndex = t
    return veriIsimleri[minIndex]

veriIsimleri, veriResimler = veriYukleme()
print(veriIsimleri)


while True:
    ret, kare = kamera.read()
    kesilenKare = kare[0:300, 0:300]
    kesilenKare_HSV = cv.cvtColor(kesilenKare, cv.COLOR_BGR2HSV)

    altDeger = np.array([108, 23, 82], dtype = "uint8")
    ustDeger = np.array([179, 255, 255], dtype = "uint8")

    renkFiltresiSonucu = cv.inRange(kesilenKare_HSV,altDeger,ustDeger)
    renkFiltresiSonucu = cv.morphologyEx(renkFiltresiSonucu, cv.MORPH_CLOSE, kernel)
    ret,renkfiltresi_sonucu_gri = cv.threshold(renkFiltresiSonucu,127,255,cv.THRESH_BINARY_INV)
    

    sonuc = kesilenKare.copy()
    toplam = np.zeros((80,170,3), np.uint8)#100,160

    contours, hierarchy = cv2.findContours(renkfiltresi_sonucu_gri, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    maxGenislik = 0
    maxUzunluk = 0
    maxIndex = -1

    for t in range(len(contours)):
        cnt = contours[t]
        x,y,w,h = cv.boundingRect(cnt)
        if(w>maxGenislik and h>maxUzunluk):
            maxUzunluk = h
            maxGenislik = w 
            MaxIndex = t

    if(len(contours)>0):
        x,y,w,h = cv.boundingRect(contours[maxIndex])
        esit = cv.rectangle(sonuc, (x,y), (x+w, y+h),(0,0,0),3)
        elFiltresi = renkfiltresi_sonucu_gri[y:y+h, x:x+w]
        elFiltresi = cv2.bilateralFilter(elFiltresi, 5, 50, 100)
        
        
        sayi1 = siniflandir(elFiltresi,veriIsimleri,veriResimler)      
        cv2.putText(toplam, str(sayi1)+"+", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        sayi2="5"
        #sayi2 = siniflandir(elFiltresi,veriIsimleri,veriResimler)
        cv2.putText(toplam, str(sayi2)+"=", (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        sayi3 = int(sayi1)+int(sayi2)
        sayi3=str(sayi3)
        cv2.putText(toplam, sayi3, (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        

        sayi1 = siniflandir(elFiltresi,veriIsimleri,veriResimler)
        cv2.putText(sonuc, sayi1, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv.imshow("EL RESMI",elFiltresi)
        #print(siniflandir(elFiltresi,veriIsimleri,veriResimler))


    cv.imshow("KARE",kare)
    cv.imshow("SONUC",sonuc)
    cv.imshow("TOPLAMA ISLEMLERI",toplam)

    if cv.waitKey(37) == ord('q'):
            break

kamera.release()
cv.destroyAllWindows()
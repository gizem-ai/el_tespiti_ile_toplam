import os
import cv2
import cv2 as cv
import numpy as np
from numpy.core.getlimits import MachArLike

kamera = cv.VideoCapture(0)
kernel = np.ones((12,12),np.uint0)

#öncelikbusayfa
sayi ="0" #tamamdır
#sayi ="1" #tamamdır
#sayi ="2" #tamamdır
#sayi ="3" #tamamdır
#sayi ="4" #tamamdır
#sayi ="5"# tamamdır
#sayi ="6" tamamdır
#sayi ="7" #tamamdır
#sayi ="8" #tamamdır
#sayi ="9" tamamdır
#sayi = "Teşekkürler" #sonraya sal

while True:
    ret, kare = kamera.read()
    kesilenKare = kare[0:275, 0:275]
    kesilenKare_HSV = cv.cvtColor(kesilenKare, cv.COLOR_BGR2HSV)

    altDeger = np.array([108, 23, 82], dtype = "uint8")
    ustDeger = np.array([179, 255, 255], dtype = "uint8")

    renkFiltresiSonucu = cv.inRange(kesilenKare_HSV,altDeger,ustDeger)
    renkFiltresiSonucu = cv.morphologyEx(renkFiltresiSonucu, cv.MORPH_CLOSE, kernel)

    sonuc = kesilenKare.copy()

    #(cnts,_) = cv.findContours(renkFiltresiSonucu, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(renkFiltresiSonucu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
        cv.rectangle(sonuc, (x,y), (x+w, y+h),(255,0,211),2)
        elFiltresi = renkFiltresiSonucu[y:y+h, x:x+w]
        elFiltresi = cv2.bilateralFilter(elFiltresi, 5, 50, 100)  # yumuşatma filtresi
        cv.imshow("EL RESMI",elFiltresi)

    cv.imshow("KARE",kare)
    cv.imshow("KESILMIS KARE",kesilenKare)
    cv.imshow("RENK FILTRESININ SONUCU",renkFiltresiSonucu)
    cv.imshow("SONUC",sonuc)

    if cv.waitKey(100) == ord('q'):
        cv.imwrite("tanimlanan/"+sayi+".jpg",elFiltresi)     #q ile çıkış yapıldığı zaman isim.jpg olarak klasör içine kayıt sağladım
        break

kamera.release()
cv.destroyAllWindows()

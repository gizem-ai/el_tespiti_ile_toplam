import cv2 as cv
import numpy as np
import os
import cv2
import random
from datetime import datetime       # pip install datatime 

"""
201613709028 - Gizem SOLMAZ
"""


sayi2=0
second=datetime.now()
kamera = cv.VideoCapture(0) #0 indexli kamerayı atadım
kernel = np.ones((12,12),np.uint0) #yumuşatma değerleri

def ResimFarkiBul (resim1,resim2): #
    resim2 = cv.resize(resim2,(resim1.shape[1],resim1.shape[0])) #kaydedilen resimlerle kameradan gelen tanımlanmış enini ve boyunu eşitler
    farkResim = cv.absdiff(resim1,resim2)  #arka planlar temizlenir
    farkSayi = cv.countNonZero(farkResim)   #sıfır olmayanları sayıyoruz
    return farkSayi

def veriYukleme():
    veriIsimleri = []
    veriResimler = []
    dosyalar = os.listdir("tanimlanan/")    #tanimlanan klasörü altında ki bütün verileri listeler
    for dosya in dosyalar:  #dosya değişkenini tanimlanan klasöirm içinde gezdirmek için for kullandım
        veriIsimleri.append(dosya.replace(".jpg",""))    #dosyada ki verilerin isimlerini .jpg yerine " " ile ismini değiştirdim/uzantıyı yazmasını engelledim
        veriResimler.append(cv.imread("tanimlanan/"+dosya,0))
    return veriIsimleri, veriResimler    #bu işlemi return ettirdim

#kameradan gelen görüntünün dosyada ki görüntülerle eş olup olmadığını kontrol eder.
def siniflandir(resim,veriIsimleri,veriResimler):  
    minIndex = 0    #doğruluk için en minimum değeri aldık
    minDeger = ResimFarkiBul(resim,veriResimler[0])    
    for t in range(len(veriIsimleri)):    #verinin uzunluğu kadar dönecek
        farkDeger = ResimFarkiBul(resim,veriResimler[t])  #kameradakiyle dosya içinde ki görüntülerin farkları farkDegere atanır
        if(farkDeger<minDeger):  #görüntüler arasında ki fark minDegerden küçük ise;
            minDeger = farkDeger  #bunlar birbirlerine eşit olurlar
            minIndex = t  
    return veriIsimleri[minIndex]   #sonuç yazdırılır
veriIsimleri, veriResimler = veriYukleme()

while True:
    ret, kare = kamera.read()
    kesilenKare = kare[0:300, 0:300] #kameradan gelen görüntüden 300x300 bir parça alınır
    kesilenKare_HSV = cv.cvtColor(kesilenKare, cv.COLOR_BGR2HSV) #alınan parça hsv ye dönüştürülür

    #ten rengi ayrımı için bir çok aralık denedim fakat en uygunun bu olduğu gördüm
    altDeger = np.array([108, 23, 82], dtype = "uint8") #ten rengi ayrımı için alt ve 
    ustDeger = np.array([179, 255, 255], dtype = "uint8")   #üst değerler  atadım
    renkFiltresiSonucu = cv.inRange(kesilenKare_HSV,altDeger,ustDeger)  #alt ve üst değerlere göre maskeleme yaptık
    renkFiltresiSonucu = cv.morphologyEx(renkFiltresiSonucu, cv.MORPH_CLOSE, kernel)   #kernel ile bulanıklaştırma işlemi yaptık ve renkFiltresiSonucu'e eşitledik
    
    #ret,renkfiltresi_sonucu_gri = cv.threshold(renkFiltresiSonucu,127,255,cv.THRESH_BINARY_INV)
    sonuc = kesilenKare.copy()  #tanımlama işlemini görüntülemek için sonuc adında yeni pencere oluşturuldu 
    toplam = np.zeros((80,170,3), np.uint8) #100,160    #toplama işlemini yaptıracağım toplam adında yeni pencere oluşturdum
    
    contours, hierarchy = cv2.findContours(renkFiltresiSonucu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    #en uygun filtrelenmiş görüntü bulunur
    #ihtiyacımız olan değişkenler yazılır
    maxGenislik = 0
    maxUzunluk = 0
    maxIndex = -1
    for t in range(len(contours)):  #kameradaki görüntüde en büyük kareyi alabilmek için 
        cnt = contours[t]
        x,y,w,h = cv.boundingRect(cnt)
        if(w>maxGenislik and h>maxUzunluk):  
            maxUzunluk = h
            maxGenislik = w 
            MaxIndex = t    
   
    if(len(contours)>0):   #contours sayısı 0 dan büyükse
        x,y,w,h = cv.boundingRect(contours[maxIndex])
        esit = cv.rectangle(sonuc, (x,y), (x+w, y+h),(0,0,0),3)  #
        elFiltresi = renkFiltresiSonucu[y:y+h, x:x+w]   #sadece tanımlanan eli görüntüden çıkartır
        elFiltresi = cv2.bilateralFilter(elFiltresi, 5, 50, 100)    #
        sayi1 = siniflandir(elFiltresi,veriIsimleri,veriResimler)    #siniflandir fonksiyonuna el filtresini atarak kamerada ki veriyi tanımlar 
        cv2.putText(sonuc, sayi1, (x,y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255,0,255), 1)      ##tanımlanan sayi1 i puttext ile sonuc penceresine yazdırır.
        cv.imshow("EL RESMI",elFiltresi)
        #print(siniflandir(elFiltresi,veriIsimleri,veriResimler))   #terminale yazdırır
        sayi1 = siniflandir(elFiltresi,veriIsimleri,veriResimler)     #siniflandir fonksiyonuna el filtresini atarak kamerada ki veriyi tanımlar 
        cv2.putText(toplam, str(sayi1)+"+", (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0,0,255), 1)  #tanımlanan sayi1 i puttext ile toplam penceresine yazdırır.
        
        first=datetime.now()    
        time=(first-second).seconds     
        if(time==7):    #her 7 saniyede bir
            second=datetime.now() 
            time=0
            sayi2=random.randint(1,11)  #random bir sayı üretilir
       
        cv2.putText(toplam, str(sayi2)+"=", (70, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0,0,255), 1)  #üretilen random sayı toplam penceresinde 2. sayı olarak yer alır
        sayi3 = int(sayi1)+int(sayi2) #kameradan tanımlanan ilk sayi ile random gelen 2. sayıyı topladım
        sayi3=str(sayi3)    #stringe dönüştürdüm çünkü puttype a str değer yazılması gerekiyor
        cv2.putText(toplam, sayi3, (110, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0,0,255), 1)   #toplama işlemi toplam penceresinde toplam sonucu olarak yerini alır 
    
    cv.imshow("hsv",kesilenKare)
    cv.imshow("KARE",kare)  #kameradan gelen görüntü
    cv.imshow("SONUC",sonuc)     #tespitin yapıldığı ekran
    cv.imshow("TOPLAMA ISLEMLERI",toplam)   #toplama işleminin yapıldığı ekran
    
    if cv.waitKey(37) == ord('q'):  #q ile çıkış
            break
kamera.release()
cv.destroyAllWindows()
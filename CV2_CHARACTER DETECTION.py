import cv2
import pytesseract
import pyttsx3

engine = pyttsx3.init()
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('TEXT.jpg')            #PUT IMAGE PATH
imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(pytesseract.image_to_string(img))

hImg,wImg=imgGRAY.shape 
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(' ')
    # print(b)
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,255,0),1)
    cv2.putText(img,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
engine.say(pytesseract.image_to_string(img))
engine.runAndWait()

cv2.imshow('Result',img)
cv2.waitKey(0)
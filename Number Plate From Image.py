import pytesseract
import shutil
import os
import scipy.ndimage as ndimage
import random
try:
 from PIL import Image
except ImportError:
 import Image
import cv2
import re
import numpy as np
count=0
trigger= True
if(trigger):

  cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
  frame = cv2.imread('car6.jpg')

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  NumberPlates = cascade.detectMultiScale(gray, 1.2, 4)
  for(x, y, w, h) in NumberPlates:
      cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
      
      cv2.putText(frame, "number plate", (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
      cropped = frame[y:y+h, x:x+w]
      
  
  cv2.imwrite('crop.jpeg',cropped)

  img1 = np.array(Image.open('crop.jpeg'))
  #img2= cv2.bilateralFilter(img1,9,75,75)
  #img2= cv2.GaussianBlur(img1,(5,5),cv2.BORDER_DEFAULT)
  img2= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  blurred_img=ndimage.gaussian_filter(img2,sigma=1.5) #Gaussian filter
  sharpened_img=ndimage.laplace(img2) # laplacian filtered image
  sharpened_img2=ndimage.laplace(blurred_img) # laplacian on gaussian
  sharpened_img3=img2-blurred_img # original minus gaussian


  cv2.imshow('window',img1) #cv2.imshow('name',file) python

  text = pytesseract.image_to_string(img2)
  text=text.upper()
  print(text)
  cv2.waitKey(0)    
  cv2.destroyAllWindows()

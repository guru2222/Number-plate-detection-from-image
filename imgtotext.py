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
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
img1 = np.array(Image.open('car6.jpg'))
#cv2.imshow('window',img1)
gray = cv2.bilateralFilter(img1, 11, 17, 17) #Blur to reduce noise
cv2.imshow('x',gray)
#edged = cv2.Canny(img1, 25, 175) #Perform Edge detection
#cv2.imshow('y',edged)
text = pytesseract.image_to_string(gray)
text=text.upper()
print(text)
cv2.waitKey(0)    
cv2.destroyAllWindows()
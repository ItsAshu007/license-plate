import cv2
import pytesseract
import re
import numpy as np
from skimage import exposure

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

harcascade = "model/numberplate_haarcade.xml"

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

min_area = 500
count = 0

while True:
    success, img = cap.read()
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)
            
            # Crop the license plate image
            img_roi = img[y: y+h, x: x+w]
            cv2.imshow("ROI", img_roi)

            # Enhance the license plate image
            enhanced_img = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            
            # Contrast stretching
            p2, p98 = np.percentile(enhanced_img, (2, 98))
            enhanced_img = exposure.rescale_intensity(enhanced_img, in_range=(p2, p98))
            
            # Histogram equalization
            enhanced_img = cv2.equalizeHist(enhanced_img)
            
            # # Adaptive thresholding
            # enhanced_img = cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Use pytesseract to extract text from the enhanced license plate image
            text = pytesseract.image_to_string(enhanced_img)
            
            # Filter out irrelevant characters using regex
            license_plate_text = re.sub('[^A-Z0-9]', '', text)
            
            print(license_plate_text)

    cv2.imshow("Result", img)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

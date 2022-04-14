import re
import cv2
import imutils
import easyocr
import numpy as np
from matplotlib import pyplot as pp
from pytesseract import pytesseract

class Detect:

    # ___Constructor____
    def __init__(self, image):
        self.mask_image = None
        self.tessertact_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        originalImage = self.read_image(image)
        self.showImage('Original Image',originalImage,True)
        grayImage = self.applyGrayScale(originalImage)
        self.showImage('Gray Image',grayImage,True)
        bFilter = self.applyBiFilter(grayImage)
        self.showImage('BiLiteral Filtering',bFilter,True)
        edgeFilter = self.applyEdgeFilter(bFilter)
        self.showImage('Edge Filtering',edgeFilter,True)
        contour = self.findContour(edgeFilter,0)
        location = self.findPolygons(contour)
        print(f'Contour Points: {location}')
        maskedImage = self.maskImage(originalImage,grayImage,location)
        self.pyPlot(maskedImage,True)
        numPlate = self.rectImage(originalImage,location)
        croppedImage = self.cropImage(originalImage)
        text = self.extract_character(None,croppedImage)
        self.showImage('Number Plate',numPlate,True)
        print(text)
        res = self.drawRectangle(location, originalImage, text)
        self.showImage('Number Plate', res, True)

    # Load Image
    def read_image(self, img_path):
        return cv2.imread(img_path)

    # Convert Image to Grayscale
    def applyGrayScale(self, img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # Reduce Image Noise, Smoothen Image
    def applyBiFilter(self, img):
        return cv2.bilateralFilter(img,11,17,17)

    # Create edging on grayscale image
    def applyEdgeFilter(self, img):
        return cv2.Canny(img,10,200)

    # Find Contour lines using edging
    def findContour(self, img, resultLength):
        contourPoints = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contourPoints)
        if resultLength:
            contours = sorted(contours, key=cv2.contourArea,reverse=True)[:resultLength]
        else:
            contours = sorted(contours, key=cv2.contourArea,reverse=True)

        return contours

    # Find Polygon Location from contour lines
    def findPolygons(self, contours):
        loc = None
        for contour in contours:
            approxPolygon = cv2.approxPolyDP(contour,10, True)
            if len(approxPolygon) == 4:
                loc = approxPolygon
                break
        return loc

    # Masking Image Location
    def maskImage(self, original_img, gray_img, location):
        self.mask_image = np.zeros(gray_img.shape,np.uint8)
        new_image = cv2.drawContours(self.mask_image,[location],0,255,-1)
        new_image = cv2.bitwise_and(original_img,original_img,mask=self.mask_image)
        return new_image

    # Rectangle around the location
    def rectImage(self, img, location):
        return cv2.rectangle(img, tuple(location[0][0]),tuple(location[2][0]),(0,255,0),3)
    
    # Crop image 
    def cropImage(self, img):
        (x, y) = np.where(self.mask_image==255)
        (xMin, yMin) = (np.min(x), np.min(y))
        (xMax, yMax) = (np.max(x), np.max(y))
        cropped_image = img[xMin:xMax+1, yMin:yMax+1]

        return cropped_image

    # show image
    def showImage(self, title, data, wait=False):
        cv2.imshow(title, data)
        if wait:
            cv2.waitKey(0)
    
    # plot image on pyplot
    def pyPlot(self, img, wait=False):
        pp.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if wait:
            pp.waitforbuttonpress()
    
    # write text around rectangle
    def drawRectangle(self, location, image, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        return cv2.putText(image, text=text, org=(location[0][0][0]-90, location[1][0][1]+90), fontFace=font, fontScale=1, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)
    
    # extract characters from image
    def extract_character(self, lib, img):
        if lib == 'tesseract-ocr':
            pytesseract.tesseract_cmd = self.tessertact_path
            result = self.remove_special_characters(pytesseract.image_to_string(img))
        if lib == 'easy-ocr':
            reader = easyocr.Reader(['en'])
            text = reader.readtext(img)
            result = self.remove_special_characters(text[0][-2])
        else:
            pytesseract.tesseract_cmd = self.tessertact_path
            result = f'Tesseract OCR: {self.remove_special_characters(pytesseract.image_to_string(img))}'
            reader = easyocr.Reader(['en'])
            text = reader.readtext(img)
            result = f'Easy OCR: {self.remove_special_characters(text[0][-2])} \n{result}'

        return result

    # remove special character from result
    def remove_special_characters(self, text):
        pattern = r'[^A-Za-z0-9]+'
        return re.sub(pattern,'',text)


if __name__ == "__main__":
    
    images = [
        'samples/plate1.jpg',
        'samples/plate2.jpg',
        'samples/plate3.png',
        'samples/plate4.jpg',
        'samples/plate5.jpg',
        'samples/plate6.png',
        'samples/plate7.jpg',
        'samples/plate8.jpg',
        'samples/plate9.jpg',
        'samples/plate10.jpg',
        'samples/plate11.jpg',
        'samples/plate12.jpeg'
    ]
    Detect(images[0])

    
    


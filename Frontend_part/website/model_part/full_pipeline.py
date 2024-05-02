#Importing necessary libraries
import cv2

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings('ignore')


import cv2
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import pandas as pd 
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from selenium.webdriver import Chrome,ChromeOptions,ChromeService 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import time





model=tf.keras.models.load_model(r'C:\Users\megha\Downloads\tamil-recognition-main\tamil-recognition-main\megha\website\model_part\model_trained\Image_model.keras')
model.compiled_metrics == None
CATEGORIES=pd.read_csv(r'C:\Users\megha\Downloads\tamil-recognition-main\tamil-recognition-main\megha\website\model_part\Category.csv')
CATEGORIES=list(CATEGORIES['Category'])

def perform_segemenation(image_path):
    print('segmentation')
    def correct_skew(image, delta=1, limit=5):
        def determine_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
            return histogram, score

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                borderMode=cv2.BORDER_REPLICATE)

        return best_angle, rotated
    
    image = cv2.imread(image_path)
    angle, rotated = correct_skew(image)

    # cv2.imwrite('rotated-2.jpg', rotated)
    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)

    dilate = cv2.dilate(thresh1, None, iterations=1)

    cnts,h = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)


    contours_with_positions = []

    # Loop through contours and store their positions
    for contour in cnts:
        # Get the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Store the contour along with its position
        contours_with_positions.append((x, y, contour))

    # Sort contours based on their x-coordinate
    contours_with_positions.sort(key=lambda x: x[0])

    orig = image.copy()
    i = 0
    for cnt in contours_with_positions:
        cnt=cnt[2]
        
        # Check the area of contour, if it is very small ignore it
        if(cv2.contourArea(cnt) < 200):
            continue

        # Filtered countours are detected
        x,y,w,h = cv2.boundingRect(cnt)

        # Taking ROI of the cotour
        roi = image[y:y+h+5, x:x+w+5]

        # Mark them on the image if you want
        cv2.rectangle(orig,(x,y),(x+w+5,y+h+5),(0,255,0),2)

        # Save your contours or characters
        cv2.imwrite(f"website\\model_part\\segmented_images\\{i}.png",roi)

        i = i + 1
        print('saved')
    cv2.imwrite(r"C:\Users\megha\Downloads\tamil-recognition-main\tamil-recognition-main\megha\website\static\box\box.png",orig)


def translate_tamil(tamil_word):
    chromedriver_path=r'C:\Users\megha\Downloads\tamil-recognition-main\tamil-recognition-main\megha\website\model_part\chromedriver.exe'

    service=ChromeService(executable_path=chromedriver_path)
    options=ChromeOptions()
    options.add_argument('--headless')


    driver=Chrome(service=service,options=options)

    driver.get('https://www.google.com/search?q=tamil+to+english+translation+&sca_esv=6246709ce63302d6&sxsrf=ACQVn09yPWI52zUAycVcpx4W4KUs7_tw9g%3A1711975792736&ei=cK0KZoDMLO2TseMP-NqIoAY&ved=0ahUKEwiAxPXnhqGFAxXtSWwGHXgtAmQQ4dUDCBA&uact=5&oq=tamil+to+english+translation+&gs_lp=Egxnd3Mtd2l6LXNlcnAiHXRhbWlsIHRvIGVuZ2xpc2ggdHJhbnNsYXRpb24gMgQQIxgnMgsQABiABBiKBRiRAjILEAAYgAQYigUYkQIyBRAAGIAEMg4QABiABBiKBRiRAhixAzILEAAYgAQYigUYkQIyChAAGIAEGBQYhwIyBRAAGIAEMgUQABiABDIFEAAYgARI7wZQrANYrANwAXgBkAEAmAGkAaABpAGqAQMwLjG4AQPIAQD4AQGYAgKgAqwBwgIKEAAYRxjWBBiwA5gDAIgGAZAGBJIHAzEuMaAHgQg&sclient=gws-wiz-serp')

    input_box=driver.find_elements(By.TAG_NAME,"textarea")
    input_box=input_box[1]
    input_box.send_keys(tamil_word)

    time.sleep(3)

    output=driver.find_elements(By.XPATH,"//span[@class='Y2IQFc']")
    output=output[2].text

    # time.sleep(10)

    return output


def prediction_part(PREDICTION_PATH):
    

    def preprocess(image_path):

        numpy_img=cv2.imread(image_path)
        # Convert NumPy array to PIL image
        img = Image.fromarray(numpy_img)

        # Convert to grayscale
        converted = img.convert("L")

        # Invert colors
        inverted = ImageOps.invert(converted)
        thick=inverted

        # # Apply maximum filter
        thick = inverted.filter(ImageFilter.MaxFilter(5))

        # Calculate resizing ratio (adjusting for smaller size)
        ratio = 32.0 / max(thick.size)

        # Resize image
        new_size = tuple([int(round(x*ratio)) for x in thick.size])
        res = thick.resize(new_size, Image.LANCZOS)
        
        # Convert resized image back to NumPy array
        arr = np.asarray(res)

        # Calculate center of mass
        com = ndimage.center_of_mass(arr)

        # Create blank image
        result = np.zeros((64, 64), dtype=np.uint8)

        # Calculate paste box coordinates
        box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))

        # Paste resized image onto blank image
        result[box[1]:box[1]+arr.shape[0], box[0]:box[0]+arr.shape[1]] = arr

        return result
    def model_prediction(image_path):
        image=preprocess(image_path)
        
        image_reshaped=image.reshape(-1,64,64,1)

        intial_prediction=model.predict(image_reshaped,verbose=0)
        
        class_=np.argmax(intial_prediction)
        
        y_pred=CATEGORIES[class_]
        return y_pred
    

    
    
    image_path=[]
    for i in os.listdir(PREDICTION_PATH):
        path=os.path.join(PREDICTION_PATH,i)
        image_path.append(path)

    prediction=[]
    for i in image_path:
        y_pred=model_prediction(i)
        prediction.append(y_pred)
        os.remove(i)
    

    string_to_send=''
    for i in prediction:
        string_to_send+=i
        

    return string_to_send

def predict_(path):
    perform_segemenation(path)

    string_to_send=prediction_part(r'C:\Users\megha\Downloads\tamil-recognition-main\tamil-recognition-main\megha\website\model_part\segmented_images')
    print(string_to_send)
    translated=translate_tamil(string_to_send)

    return translated



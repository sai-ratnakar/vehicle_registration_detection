import os
from re import L
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import detect_plate, display,fix_dimension,find_contours,segment_characters
# import tensorflow as tf
# from sklearn.metrics import f1_score 
# from tensorflow.keras import optimizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras.models import model_from_json
import time

# plate_cascade = cv2.CascadeClassifier('models/indian_license_plate.xml')

json_file = open('models/mvrpjsonModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('models/mvrpWeights.h5')

# img = cv2.imread('License-plate-india-header-1920x730.jpg')

print('Model Loaded.....!!!')

# output_img, plate = detect_plate(img)

# char = segment_characters(plate)
country_code = ['KA','TS','AP','TN',
'TW','UP','30','29','3V','WB','W2', 'V2','BQ' ]

def show_results(char1):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c
    output = []
    for i,ch in enumerate(char1): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model 
        y_ = model.predict(img)[0] #predicting the class
        y_ = list(map(int, y_))
        ind = y_.index(1.0)
        character = dic[ind] #
        output.append(character)
        
    plate_number = ''.join(output)
    
    return plate_number

import difflib



# cap = cv2.VideoCapture('vid15.MOV')

nums=[]


def number_plate_detect(frame):
    try:
        output_img, plate = detect_plate(frame)
        char = segment_characters(plate)
        # print(char)
        plate_number = show_results(char)
        
        # print(output_list)
        if len(plate_number)>6 and plate_number not in nums and plate_number[:2] in country_code:
            nums.append(plate_number)
            output_img, plate = detect_plate(frame, plate_number)
            print("Detected a number Plate ===>",plate_number)
            # time.sleep(0.5)
        output_img = cv2.resize(output_img, (960, 540)) 
        # cv2.imshow('Vehicle plate Recognition',output_img)
        ret, jpeg = cv2.imencode('.jpeg', output_img)
        return jpeg.tobytes()
        
    except Exception as e:
         
        pass
        
        frame = cv2.resize(frame, (960, 540))
         # cv2.imshow('Vehicle plate Recognition',frame)
         
        ret, jpeg = cv2.imencode('.jpeg', frame)
        return jpeg.tobytes()
    
        

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


# cap.release()
# cv2.destroyAllWindows()  

  
# print(show_results())
# plate_number = show_results()
# output_img, plate = detect_plate(img, plate_number)
# display(output_img, 'detected license plate number in the input image')


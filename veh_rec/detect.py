import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
country_code = ['KA','TS','AP','TN',
'TW','UP','30','29','3V','WB','W2', 'V2','BQ' ]
import cv2
#import difflib
from utils import detect_plate, fix_dimension,segment_characters
from tensorflow.keras.models import model_from_json


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


cap = cv2.VideoCapture('video15.mp4')

nums=[]
# prev_plate=''
i=0
#   and plate_number[:2] in country_code
while(cap.isOpened()):

    ret, frame = cap.read()     
    if ret == False:
        break
    try:
       if i%5==0:
            output_img, plate = detect_plate(frame)
            char = segment_characters(plate)
            plate_number = show_results(char)
            # if prev_plate:
            #     output_list = [li for li in difflib.ndiff(plate_number, prev_plate) if li[0] != ' ']
            # else:
            #     prev_plate = plate_number
            if len(plate_number)>6 and plate_number not in nums : ###
                nums.append(plate_number)
                # prev_plate=plate_number
                output_img, plate = detect_plate(frame, plate_number)
                print("Detected a number Plate ===>",plate_number)
            
            output_img = cv2.resize(output_img, (960, 540)) 
            cv2.imshow('Vehicle plate Recognition',output_img)
        
    except :
        pass
        
        frame = cv2.resize(frame, (960, 540)) 
        cv2.imshow('Vehicle plate Recognition',frame)
        
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()  

  
# print(show_results())
# plate_number = show_results()
# output_img, plate = detect_plate(img, plate_number)
# display(output_img, 'detected license plate number in the input image')


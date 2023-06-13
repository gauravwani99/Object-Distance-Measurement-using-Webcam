
import cv2 # importing computer vision 2 library
import pandas as pd # importing pandas library
import numpy as np # importing numpy library
import matplotlib.pyplot as plt # importing matplotlib library for making graphs 
from sklearn import linear_model # importing linear_model from sk_learn library


df = pd.read_csv("finalsheet.csv") # converting a the csv file into a dataframe
# this csv file consist of all the areas of contour taken at respective distances (10 entries)
'''plotting the graph area vs distance graph''' 
plt.xlabel('area') # x-axis label (independent variable)
plt.ylabel('distance') # y-axis label (dependent variable)
plt.scatter(df.area, df.distance, color='red', marker='+')
#this will mark the points on the coordinate axis
#the sign will we of '+' and the colour of the sign is red
reg = linear_model.LinearRegression()  #implementation of sklearn library
#we will be using linear regression in our data
reg.fit(df[['area']], df.distance) #this will accept an input and then give the output after analyzing data already present

distance = 0 # this is the variable used for storing the distance between 
video = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN  # Font style for writing text on video frame
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Setting the camera resolution (width of the screen) (1960 pixels)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Setting the camera resolution (height of the screen) (1080 pixels)
Kernal = np.ones((3, 3), np.uint8)  # this will produce an array of 3 entries of 8 bit entries

while 1: # running an infinite loop
    ret, frame = video.read()  # Read image frame
    frame1 = cv2.flip(frame, +1)  # Mirror image frame
    if not ret:  # If frame is not read then exit the code 
        break
    if cv2.waitKey(1) == ord('s'):  # Press s to stop the code
        break
    frame2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)  # converting the captured frame from BGR to HSV
    lb = np.array([0, 0, 10])  # lower bound of the HSV values to be detected
    ub = np.array([255, 255, 45])  # upper bound of the HSV values to be detected

    mask = cv2.inRange(frame2, lb, ub)  # create a mask of the colours detected
    cv2.imshow('Masked Image', mask) # showing the masked image 

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernal)  # remove the unwanted detected pixels around the object detected
    cv2.imshow('Opening', opening) # showing the resulted image

    res = cv2.bitwise_and(frame1, frame2, mask=opening)  # colours to be detected are applied on the mask
    cv2.imshow('Resulting Image', res) # showing the resulted image

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # to find the contours
    #  Contours is a Python list of all the contours in the image
    if len(contours) != 0: # if there exist any contour
        cnt = contours[0] # getting the countour for the object 
        area = cv2.contourArea(cnt)  # area of the contour
        if area > 100: # if the area id greter than 100
            S = 'Area of contour: ' + str(area) # it gives the area of countour 
            cv2.putText(frame1, S, (5, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA) # printing the area of countour on resulting image
            a = reg.predict([[area]]) # predicting the distance for an input area
            t = 'Distance Of Object: ' + str(a) # it gives the distance of object from the webcam
            cv2.putText(frame1, t, (5, 100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawContours(frame1, cnt, -1, (0, 255, 0), 3)  # to draw the contour for the object
    cv2.imshow('Original Image', frame1)  # showing the resulting image

video.release()  # Release memory
cv2.destroyAllWindows()  # Close all the windows
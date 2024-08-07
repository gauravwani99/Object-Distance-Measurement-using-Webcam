This project intends to find the distance of an object by webcam using the OpenCV library. The principle of the project is
that the contour area decreases as the distance from the camera increases. We first convert our video frame to HSV and then put the
range of the object's colors to be detected. On detection, it will start recording the area of contours. The obtained contour area and
the measured distances will be stored in a dataframe. The OpenCV library of python is the most widely used tool for image
processing. It is used in tracking moving objects, face detection, object disclosure, etc. We are just using it here for distance
estimation.

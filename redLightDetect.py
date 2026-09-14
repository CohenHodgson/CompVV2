'''
next step is NEURAL NET
'''

import numpy as np
import cv2 as cv
import glob # for automated file detect
import math # probably don't need this

'''
how to find red light from image.

1. get image
2. get colors of image (through cv.inRange mask)
3. cv.imshow mask
4. filter mask, binary (ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY))
5. get contours of colored shape
6. draw rectangle around position?
7. return "red light detected," then add neural detection
'''

img = cv.imread(str(input())) # put full file name into terminal (including extension)

assert img is not None, "file could not be read"

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # scan image and convert BGR to hsv


lower_red = np.array([0,100,100]) # in hsv values which are weird 
higher_red = np.array([10,255,255])

mask = cv.inRange(hsv, lower_red, higher_red)

kernel = np.ones((3,3), np.uint8) # 3,3 = anything smaller than 3x3px is removed.
# 8 bits, because 255 bgr.

noise_removed = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 3) 
# takes mask, applies erosion followed by dilation under the directions of kernel 3 time.


contours, hierachy = cv.findContours(noise_removed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

image_contours = cv.drawContours(img, contours, -1, (255,255,255), 3)

'''
find white pixels, group them and find their center, then draw
rectangle around said center. 
'''

y, x = np.where(np.all(image_contours == [255,255,255], axis = -1)) # you just need two input values for np.all()? axis = -1 means nothing in this context
# find all white from noise_removed, give index nums of positions.

# avg y,x? max and min y and x for radius. max and min x is diameter?

if len(x) > 0: # if white found
    diameter = max(x) - min(x)
    radius = math.ceil(diameter / 2)
    center_x = int(np.mean(x))
    center_y = int(np.mean(y))
    cv.circle(image_contours, (center_x, center_y), radius=(radius + 10), color=(0, 255, 0), thickness=2) # radius + 10 so it isn't super tight
    bottom_left = (center_x - 150), (center_y - 50)
    cv.putText(image_contours, "Red Light", bottom_left, cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0)) # 0.5 = font size


cv.imshow("Display", img)


cv.waitKey(5000)




'''
Neural net below:



'''
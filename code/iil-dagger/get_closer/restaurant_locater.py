import numpy as np
import argparse
import imutils
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class restaurant_localizer():
    def __init__(self, templateGrays):
        self.templateGrays

def select_yellow(image):
    converted = convert_hls(image)
    # yellow color mask
    lower = np.uint8([ 10,   0, 90])
    upper = np.uint8([ 30, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)

    return cv2.bitwise_and(image, image, mask = yellow_mask)
def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def get_best_match(gray, templateGrays, threshold=0):
    found = None
    res=None
    for template in templateGrays:
        (tH, tW) = template.shape[:2]
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r, tH, tW)

    (maxVal, maxLoc, r, tH, tW) = found
    if maxVal<=threshold:
        return None,None,None,None
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    return startX,startY,endX,endY

def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_template_grays(templates):
    templateGrays=[]
    for template in templates:
        if template is None:
            continue
        templateYellow=select_yellow(template)
        templateGray=cv2.cvtColor(templateYellow, cv2.COLOR_BGR2GRAY)
        templateGray = cv2.Canny(templateGray, 50, 200)
#         cv2.imshow("template", templateGray)
#         cv2.waitKey(0)
        templateGrays.append(templateGray)
    return templateGrays

def crop_image(image,startX,startY,endX,endY):
    return image[startY:endY,startX:endX]

def find_duck(image, duck_template_grays):
    startX,startY,endX,endY=get_best_match(cv2.GaussianBlur(get_gray(select_yellow(image)), (5, 5), 0),duck_template_grays,0.4)
    if startX is None:
        return None,None,None,None
    return startX,startY,endX,endY




import csv
from random import shuffle

import numpy as np
import pickle
import re
import os
from PIL import Image

DATA_ANNOTATIONS = "../data/allAnnotations.csv"
IDX_DICT = {"FileName":0,
            "AnnotationTag":1,
            "UpperLeftX": 2,
            "UpperLeftY": 3,
            "LowerRightX": 4,
            "LowerRightY": 5,
            }

RESIZE_IMAGE = True  # resize the images and write to 'resized_images/'
GRAYSCALE = True  # convert image to grayscale? this option is only valid if RESIZE_IMAGE==True (FIXME)
TARGET_W, TARGET_H = 400, 260  # 1.74 is weighted avg ratio, but 1.65 aspect ratio is close enough (1.65 was for stop signs)

if(1):
    sign_map = {'stop': 1, 'pedestrianCrossing': 2}  # only 2 sign classes (background class is 0)
else:
    sign_map = {}  # sign_name -> integer_label
    with open('signnames.csv', 'r') as f:
        for line in f:
            line = line[:-1]  # strip newline at the end
            integer_label, sign_name = line.split(',')
            sign_map[sign_name] = int(integer_label)

def readAnnotation(fileName):
    '''
    Read in CSV
    '''

    class_list = []
    box_coords_list = []

    csvList = []
    label_set = set()
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)
        for row in reader:
            sign_name = row[IDX_DICT["AnnotationTag"]]

            if sign_name != 'stop' and sign_name != 'pedestrianCrossing':
                continue  # ignore signs that are neither stop nor pedestrianCrossing signs

            sign_class = sign_map[sign_name]
            row[IDX_DICT["FileName"]] = "data/" + row[IDX_DICT["FileName"]]

            image = Image.open(row[IDX_DICT["FileName"]])
            orig_w, orig_h = image.size

            # Resize image, get rescaled box coordinates
            box_coords = np.array([int(x) for x in row[2:6]])
            # Rescale box coordinates
            x_scale = TARGET_W / orig_w
            y_scale = TARGET_H / orig_h

            ulc_x, ulc_y, lrc_x, lrc_y = box_coords
            new_box_coords = (ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale, lrc_y * y_scale)
            new_box_coords = [round(x) for x in new_box_coords]
            box_coords = np.array(new_box_coords)


            csvList.append(row)
            label_set.add(row[IDX_DICT["AnnotationTag"]])

    shuffle(csvList)
    return csvList, label_set


def main():
    csvList, label_set = readAnnotation(DATA_ANNOTATIONS)
    print(csvList)




if __name__ == "__main__":
    main()

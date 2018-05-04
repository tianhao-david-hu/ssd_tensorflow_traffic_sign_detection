import csv
from random import shuffle

import numpy as np
import pickle
import re
import os
from PIL import Image
import ntpath
from settings import GRAYSCALE, PARTIAL_DATASET
DATA_ANNOTATIONS = "./original_data/allAnnotations.csv"
IDX_DICT = {"FileName": 0,
            "AnnotationTag": 1,
            "UpperLeftX": 2,
            "UpperLeftY": 3,
            "LowerRightX": 4,
            "LowerRightY": 5,
            }

RESIZE_IMAGE = True  # resize the images and write to 'resized_images/'
TARGET_W, TARGET_H = 400, 260  # 1.74 is weighted avg ratio, but 1.65 aspect ratio is close enough (1.65 was for stop signs)

input_root_folder = "original_data/"
output_root_folder = "processed_data/"

if (PARTIAL_DATASET==True):
    sign_map = {'stop': 1, 'pedestrianCrossing': 2}  # only 2 sign classes (background class is 0)
else:
    sign_map = {}  # sign_name -> integer_label
    with open('signnames.csv', 'r') as f:
        for line in f:
            line = line[:-1]  # strip newline at the end
            integer_label, sign_name = line.split(',')
            print(sign_name)
            sign_map[sign_name] = int(integer_label)
print(sign_map)
merged_annotations = []
with open(input_root_folder+'allAnnotations.csv', 'r') as f:
    for line in f:
        line = line[:-1]  # strip trailing newline
        merged_annotations.append(line)

image_files = []

with open(DATA_ANNOTATIONS, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader)
    for row in reader:
        # Get signname
        sign_name = row[IDX_DICT["AnnotationTag"]]
        # Skip it is not of interest
        if (PARTIAL_DATASET == True):
            bIfNeeded = False
            for name in sign_map:
                if sign_name == 'stop' or sign_name == 'pedestrianCrossing':
                #if sign_name == 'pedestrianCrossing':
                    bIfNeeded = True
                    break  # ignore signs that are neither stop nor pedestrianCrossing signs
            if (bIfNeeded != True):
                continue
        image_files.append(row[IDX_DICT["FileName"]])

# Create raw data pickle file
data_raw = {}
ii=0
for image_file in image_files:
    # Find box coordinaimage_filetes for all signs in this image
    class_list = []
    box_coords_list = []
    for line in merged_annotations:
        if(image_file[0]=='a'):
            continue
        if re.search(image_file, line):
            fields = line.split(';')

            # Get sign name and assign class label
            sign_name = fields[1]
            if (PARTIAL_DATASET == True):
                bIfNeeded = False
                for name in sign_map:
                    if sign_name == 'stop' or sign_name == 'pedestrianCrossing':
                        # if sign_name == 'pedestrianCrossing':
                        bIfNeeded = True
                        break  # ignore signs that are neither stop nor pedestrianCrossing signs
                if (bIfNeeded != True):
                    continue
            #print(image_file)
            sign_class = sign_map[sign_name]
            class_list.append(sign_class)

            # Resize image, get rescaled box coordinates
            box_coords = np.array([int(x) for x in fields[2:6]])

            original_filename = input_root_folder+ image_file
            processed_filename = output_root_folder + image_file

            if RESIZE_IMAGE:
                # Resize the images and write to 'resized_images/'
                image = Image.open(original_filename)
                orig_w, orig_h = image.size

                if GRAYSCALE:
                    image = image.convert('L')  # 8-bit grayscale
                image = image.resize((TARGET_W, TARGET_H), Image.LANCZOS)  # high-quality downsampling filter

                file_name_only=ntpath.basename(processed_filename)
                file_name_only_length=len(file_name_only)
                path_without_name=processed_filename[:-file_name_only_length]

                if not os.path.exists(path_without_name):
                    os.makedirs(path_without_name)
                image.save(processed_filename)

                # Rescale box coordinates
                x_scale = TARGET_W / orig_w
                y_scale = TARGET_H / orig_h

                ulc_x, ulc_y, lrc_x, lrc_y = box_coords
                new_box_coords = (ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale, lrc_y * y_scale)
                new_box_coords = [round(x) for x in new_box_coords]
                box_coords = np.array(new_box_coords)
                ii=ii+1
                print(ii)

            box_coords_list.append(box_coords)

    #if ii>20:
    #    break;

    if len(class_list) == 0:
        continue  # ignore images with no signs-of-interest
    class_list = np.array(class_list)
    box_coords_list = np.array(box_coords_list)

    # Create the list of dicts
    the_list = []
    for i in range(len(box_coords_list)):
        d = {'class': class_list[i], 'box_coords': box_coords_list[i]}
        the_list.append(d)



    data_raw[output_root_folder+image_file] = the_list



with open(output_root_folder+'data_raw_%dx%d.csv' % (TARGET_W, TARGET_H), 'w') as csvfile:
    fieldnames = ["Filename",
                  "Annotation tag",
                  "Upper left corner X",
                  "Upper left corner Y",
                  "Lower right corner X",
                  "Lower right corner Y"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for image_file in image_files:
        if (image_file[0] == 'a'):
            continue
        #print(data)
        the_list=data_raw[output_root_folder+image_file]
        for d in the_list:
            writer.writerow({'Filename': output_root_folder+image_file,
                             'Annotation tag': d['class'],
                             'Upper left corner X': d['box_coords'][0],
                             'Upper left corner Y': d['box_coords'][1],
                             'Lower right corner X': d['box_coords'][2],
                             'Lower right corner Y': d['box_coords'][3],})



with open(output_root_folder+'data_raw_%dx%d.p' % (TARGET_W, TARGET_H), 'wb') as f:
    pickle.dump(data_raw, f)

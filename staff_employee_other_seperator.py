# KAUSAIN'S MADE 
#  INFO not updated

"""ABOUT
1. To convert VOTT .json annotations into crops of respective dataset images
2. To visualize annotations on dataset (VOTT Format .json files)
"""
"""HOW TO USE
# python detect_and_remove.py --dataset dataset/
# python detect_and_remove.py --dataset dataset/ --remove 1
"""
"""HOW IT WORKS
1. Provide a folder with images and VOTT .json annotations and the result will be
a folder with {DATASET_ROOT}_crops is created with {DATASET_ROOT} folder, this 
folder involves all classes/tags of dataset i.e tags of .json files and within 
each class/tag folder their will be annotated crops
2. Viewing annotations on images is also possible  
"""
"""Dependencies
- imutils
- numpy
- opencv
"""

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import json
import glob
# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
	help="whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())
dict_for_crop_against_img = {}
'''dict_for_crop_against_img
{img_name_1:[[crops1, class], [crop2, class], [crop3, class] ..], img_name_2: [...], ...}
{img_name_1:[[[x,y,w,h], car], [[x,y,w,h], bus], [[x,y,w,h], truck] ..], img_name_2: [...], ...}
'''

dict_for_img_dimesnsion = {}
# image_name =  data['asset']['name'] i.e path
# New directory to write crops
d_name = args["dataset"].rstrip("/").replace("/", "_").replace(".","")
new_directory = args["dataset"] + f"{d_name}_YOLOv5_format_annotations/"
if not os.path.exists(new_directory):
	os.makedirs(new_directory)

def crop(x0,x1,y1):

def save_img(img,path):



def get_json_files(directory):
	files = glob.glob(f"{directory}/*.json")
	for file in files:
		# file_name = file.split("/")[-1]
		# print("file", file)
		with open(f'./{file}') as f:
			data = json.load(f)
			img_height = data['asset']['size']['height']
			img_width = data['asset']['size']['width']
			img_name = f"{directory}{data['asset']['name']}"
			bboxes_for_crop = plot_bboxes(img_name, data)
			dict_for_crop_against_img[img_name] = bboxes_for_crop 
			dict_for_img_dimesnsion[img_name] = {'h':img_height, 'w':img_width} 
	return dict_for_crop_against_img, dict_for_img_dimesnsion

def plot_bboxes(img_path, dict_json_bbox):
	bboxes_for_crop = []
	# all_bboxes = len(dict_json_bbox['regions'])
	for i, bbox in enumerate(dict_json_bbox['regions']):
		h = int(bbox['boundingBox']['height'])
		x = int(bbox['boundingBox']['left'])
		y =	int(bbox['boundingBox']['top'])
		w = int(bbox['boundingBox']['width'])
		# bbox_for_crop.append([x,y,w,h])
		p1 = (int(x), int(y))
		p2 = (int(x+w), int(y+h))
		# img = cv2.rectangle(img, p1, p2, (255,255,255), 2)
		bboxes_for_crop.append([[x,y,w,h], bbox['tags'][0]])
		# print("B : " ,[[x,y,w,h], bbox['tags'][0]])
		# import pdb; pdb.set_trace()
	return bboxes_for_crop

def write_txt_file(file_name, content):
	with open(f"{file_name}.txt", 'a') as f:
		# import pdb; pdb.set_trace()
		f.write(content) 

dict_for_crop_against_img, dict_for_img_dimesnsion = get_json_files(args["dataset"])
# grab the paths to all images in our input dataset directory and
# loop over our image paths
import pdb; pdb.set_trace()
# classes_dict = ["Pedestrians", "Bicycle", "Animal Driven Carts", "MotorCycle/Scooter", "Rickshaw", "Qingqi", \
# 	"Car/Taxi/Suzuki/Pickup/Bolan/Jeep/Pajero/Land Cruiser", "Hiace Wagon", "Medium Bus/FYL. Coach/Mazda Coaster", \
# 		"Bus/Large Bus/Speedo", "Loader Pickup/Shahzore Daala/Master/Foton", "Truck: 2-AXLE", "Truck: 3-AXLE", \
# 			"Truck: 4-AXLE and above", "Tractor/Trolley"]
# classes_dict = ["Pedestrians", "Bicycle", "Animal Driven Carts", "MotorCycle/Scooter", "Rickshaw", "Qingqi", 
#   "Car/Taxi/Suzuki/Pickup/Bolan/Jeep/Pajero/Land Cruiser", "Hiace Wagon", "Medium Bus/FYL. Coach/Mazda Coaster", 
#     "Bus/Large Bus/Speedo", "Loader Pickup/Shahzore Daala/Master/Foton", "Tractor/Trolley"]
'''
Indonesia classes
'''
classes_dict = ["employee", "other", "person"]
for imageName in dict_for_crop_against_img:
	for (x, y, w, h), tag in dict_for_crop_against_img[imageName]:
		img_h, img_w = dict_for_img_dimesnsion[imageName]['h'], dict_for_img_dimesnsion[imageName]['w']
		xc_norm = (x + (w/2)) / img_w
		yc_norm =  (y + (h/2)) / img_h 
		w_norm  = w / img_w
		h_norm  = h / img_h
		# NESPAK
		# # all other classes
		# if "Truck" in tag:
		# 	tag_number = 12
		# else:
		# 	tag_number = classes_dict.index(tag)
		# wheel only
		# tag_number = 0
		'''
		Indonesia
		'''
		tag_number = classes_dict.index(tag)
		content_for_file = str(tag_number) + " " + str(xc_norm) + " " \
			+ str(yc_norm) + " " + str(w_norm) + " " + str(h_norm) + "\n" 
		file_name = imageName.split("/")[-1].replace(".jpg", "")
		output_file_path = new_directory + file_name
		write_txt_file(output_file_path, content_for_file)
print(f"Done.. Files created in {new_directory}")
# FORMAT for text file (COCO) annotaions for each image 
# ->class/tag xmin ymin width height \n

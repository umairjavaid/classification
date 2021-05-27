import cv2
import numpy as np 
from person_category_classifier import get_model_inference

def get_person_cat(img):
  result = get_model_inference(img)
  return result

def read_img(img_path):
    img = cv2.imread(img_path)
    return img

def get_person_tag(img_left, img_top, img_width, img_height, img_path=None, img=None):
  #place check img_path and img both can not be empty
  if(img_path == None and img == None):
    raise Exception("img_path and img can not be empty")
    quit()
  if(img == None and img_path != None):
    img = read_img(img_path)
  if(img == None):
    raise Exception("img can not be none")
  
  patch = img[img_top:img_top+img_height,img_left:img_left+img_width,:]
  cat = get_person_cat(patch)
  return cat

import cv2
import numpy as np 
from staff_employee_classification.person_category_classifier import modelController

mc = modelController()

def get_person_cat(img):
  result = mc.get_model_inference(img)
  return result

def read_img(img_path):
    img = cv2.imread(img_path)
    return img

def get_person_tag(row_start, row_end, col_start, col_end, img, BGR=False):
  patch = img[row_start:row_end,col_start:col_end,::(1 if BGR else -1)]
  cat = get_person_cat(patch)
  return cat











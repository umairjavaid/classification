from torchvision import models
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms, datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn as nn
from torch.utils.model_zoo import load_url
import torch.nn.functional as F
import os
from PIL import Image
import cv2 as cv
from collections import Counter
from efficientnet_pytorch import EfficientNet

class modelController:
    def __init__(self):
        self.model = self.get_model()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def get_model(self):
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 2)
        model.load_state_dict(torch.load("/home/omnoai/Desktop/umAir/saved_models/resnet_cefl_binary_combined_dataset_saphire_efficientNet.pt"))
        model.cuda()
        model.eval()
        return model

    def makeImgModelReady(self, img):
        img = self.convertToPIL(img)
        img = self.data_transforms(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        return img

    def convertToPIL(self, img):
        img = Image.fromarray(img)
        return img

    def get_inference(self, result):
        _, index = torch.max(result, 1)
        index = index.cpu().numpy()[0]
        if(index == 0):
            return "employee"
        if(index == 1):
            return "person"

    def get_model_inference(self, img):
        with torch.no_grad():
            img = self.makeImgModelReady(img)
            result = self.model(img)
            result = self.get_inference(result)
            return result

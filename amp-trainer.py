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
from efficientnet_pytorch import EfficientNet
import torch.onnx

data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((150,150)),
        #transforms.RandomResizedCrop(224),
        transforms.RandomAffine(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = "/content/dataset/"
dataset = datasets.ImageFolder(data_dir,transform= data_transforms)
train_dataset, test_dataset = torch.utils.data.random_split(dataset,[700,174], generator=torch.Generator().manual_seed(0))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)
dataloaders_dict = {'train': train_loader, 'val':test_loader}

import torch, time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

def train_model(model, dataloaders, criterion, optimizer, num_epochs=32):
    since = time.time()
    device = "cuda:0"
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    global inputs_, labels_
    #image_info.next(model,save_cams =True)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                #inputs_.data = inputs.clone()
                labels = labels.to(device)
                #for sigmoid related loss
                #labels = labels.unsqueeze(1)
                #labels = labels.float()
                #print("inputs.size(): ", inputs.size())
                #print("labels.size(): ", labels.size())
                #labels_.data = labels.clone()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                #print(model(inputs))
                #print("*****************************************************************************")

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    output = model(inputs)
                    #output = output['logits']
                    loss = criterion(output, labels)
                    _, preds = torch.max(output, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

class CEFL(nn.Module):
    def __init__(self, gamma=1):
        super(CEFL, self).__init__()
        self.gamma = gamma

    def get_prob(self, input, target):
        prob = F.softmax(input, dim=-1)
        prob = prob[range(target.shape[0]), target]
        return prob

    def get_attention(self, input, target):
        prob = self.get_prob(input, target)
        prob = 1 - prob
        prob = prob ** self.gamma
        return prob

    def get_celoss(self, input, target):
        ce_loss = F.log_softmax(input, dim=1)
        ce_loss = -ce_loss[range(target.shape[0]), target]
        return ce_loss

    def forward(self, input, target):
        attn = self.get_attention(input, target)
        ce_loss = self.get_celoss(input, target)
        prob = self.get_prob(input, target)
        loss = (1-prob)*ce_loss + prob*attn*ce_loss
        return loss.mean()

cefl = CEFL(gamma=1)

def tune_model(model, loss=cefl):
    device = "cuda:0"
    model.to(device)
    params_to_update = model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.000227913316, momentum=0.9)
    criterion = loss
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=40)
    return model

model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2)
print(model)

#model = ResNetCam(Bottleneck, [3, 4, 6, 3])
#model = load_pretrained_model(model, "resnet50")
model = tune_model(model)
#saved_model_path = "/content/saved_models"
#os.chdir(saved_model_path)
torch.save(model.state_dict(), "/content/saved_models/maskClassification-efficientNet-CEFL.pt")

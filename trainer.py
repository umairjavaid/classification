
orch.nn as nn
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


# In[43]:


os.getcwd()


# In[44]:


os.chdir("/home/omnoai/Desktop/umAir/3class_updated")


# In[45]:


os.getcwd()


# In[46]:


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


# In[47]:


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# In[48]:


class ResNetCam(nn.Module):
    def __init__(self, block, layers, num_classes=3, large_feature_map=True):
        super(ResNetCam, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return cams
        return {'logits': logits}

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes, stride)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers


# In[ ]:





# In[49]:


class VggCam(nn.Module):
    def __init__(self, features, num_classes=3):
        super(VggCam, self).__init__()
        self.features = features

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)
        return {'logits': logits}


# In[50]:


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, 1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, 1)
        self.branch5x5_2 = BasicConv2d(48, 64, 5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size,
                                     stride=stride, padding=padding)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3,
                                          stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride,
                                   padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, 1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7_2 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, (7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, (1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionCam(nn.Module):
    def __init__(self, num_classes=3, large_feature_map=False):
        super(InceptionCam, self).__init__()

        self.large_feature_map = large_feature_map

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A4 = nn.Conv2d(1024, num_classes, 1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)

        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        x = F.dropout(x, 0.5, self.training)
        feat_map = self.SPG_A4(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        if return_cam:
            feature_map = feat_map.clone().detach()
            cams = feature_map[range(batch_size), labels]
            return cams

        return {'logits': logits}


# In[51]:


def load_pretrained_model(model, name):
    if name == "resnet50":
        state_dict = load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)
        state_dict = remove_layer(state_dict, 'fc')   
    elif name == "vgg16":
        state_dict = load_url('https://download.pytorch.org/models/vgg16-397923af.pth', progress=True)
    elif name == "inceptionV3":
        state_dict = load_url('https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',progress=True)
        remove_layer(state_dict, 'Mixed_7')
        remove_layer(state_dict, 'AuxLogits')
        remove_layer(state_dict, 'fc.')
    else: 
        print("the name of the model you entered does not exsist!")
        exit()
    model.load_state_dict(state_dict, strict=False)
    return model


# In[52]:


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


# In[53]:


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


# In[54]:


train_path = 'train/'
test_path = 'test/'
data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.RandomAffine(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_data = torchvision.datasets.ImageFolder(
    root=train_path,
    transform= data_transforms
)
test_data = torchvision.datasets.ImageFolder(
    root = test_path,
    transform = data_transforms
)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    num_workers=2,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=32,
    num_workers=2,
    shuffle=True
)
dataloaders_dict = {'train': train_loader, 'val':test_loader}


# In[55]:


#inputs_ = torch.tensor(1)
#labels_ = torch.tensor(1)
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
                #labels_.data = labels.clone()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    output = model(inputs,labels)
                    output = output['logits']
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


# In[56]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def get_attention(self, input, target):
        prob = F.softmax(input, dim=-1)
        prob = prob[range(target.shape[0]), target]
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
        loss = self.alpha * ce_loss * attn
        return loss.mean()

floss1 = FocalLoss(alpha=0.25, gamma=2)


# In[57]:


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


# In[58]:


def tune_model(model, loss=nn.CrossEntropyLoss()):
    device = "cuda:0"
    model.to(device)
    params_to_update = model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.000227913316, momentum=0.9)
    criterion = loss
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=20)
    return model


# In[59]:


#model = InceptionCam()
#model = load_pretrained_model(model,"inceptionV3")
#model = tune_model(model, cefl)
#saved_model_path = "/home/omnoai/Desktop/umAir"
#os.chdir(saved_model_path)
#torch.save(model.state_dict(), "saved_models/inceptionV3_cefl.pt")


# In[ ]:


model = ResNetCam(Bottleneck, [3, 4, 6, 3])
model = load_pretrained_model(model, "resnet50")
model = tune_model(model, cefl)
saved_model_path = "/home/omnoai/Desktop/umAir"
os.chdir(saved_model_path)
torch.save(model.state_dict(), "saved_models/resnet_cefl_updateddataset.pt")


# In[ ]:


# saved_model_path = "/home/omnoai/Desktop/umAir"
# torch.save(model.state_dict(), "saved_models/resnet_untrained.pt")


# In[ ]:


# print(os.getcwd())


# In[ ]:





# In[ ]:


# model = ResNetCam(Bottleneck, [3, 4, 6, 3])
# model = load_pretrained_model(model, "resnet50")
# os.chdir("../Muaz/invalidFrames/20th feb annotated data/scripts/PREV/3class/")
# model = tune_model(model, floss1)
# os.chdir(saved_model_path)
# torch.save(model.state_dict(), "saved_models/resnet_fl.pt")


# In[ ]:





# In[ ]:


# model = ResNetCam(Bottleneck, [3, 4, 6, 3])
# model = load_pretrained_model(model, "resnet50")
# os.chdir("../Muaz/invalidFrames/20th feb annotated data/scripts/PREV/3class/")
# resnet_ce = tune_model(model)
# os.chdir(saved_model_path)
# torch.save(resnet_ce.state_dict(), "saved_models/resnet_ce.pt")

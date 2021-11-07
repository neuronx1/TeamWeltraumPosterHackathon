import torch
import matplotlib.pyplot as plt
import time
import os
import copy
import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
plt.ion()   # interactive mode
from PIL import Image
# Data augmentation and normalization for training
# Just normalization for validation
transforms_eval= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset_sorted'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def decide(path):
    model = torch.load(path)
    model.eval()
    img=Image.open("/home/florian/Desktop/hackathon2021/dataset_sorted/train/100/100 (1).jpg")
    img_tensor=transforms_eval(img)
    img_tensor.unsqueeze_(0)
    img_tensor=img_tensor.to(device)
    outputs = model(img_tensor)
    _, preds = torch.max(outputs, 1)
    print(preds)
    
#decide('checkpoints/model90.pth')   
def visualize_model(checkpointpath, input_img):
    model = torch.load(checkpointpath)
    model.eval()

    with torch.no_grad():
        img=Image.open(input_img)
        img_tensor=transforms_eval(img)
        img_tensor.unsqueeze_(0)
        img_tensor=img_tensor.to(device)
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)

        for j in range(img_tensor.size()[0]):
            #ax = plt.subplot(num_images//2, 2, images_so_far)
            #ax.axis('off')
            print('predicted: {}'.format(class_names[preds[j]]))
            #imshow(inputs.cpu().data[j])
                
                
visualize_model('checkpoints/model90.pth', "/home/florian/Desktop/hackathon2021/dataset_sorted/train/100/100 (1).jpg")
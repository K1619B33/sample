%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import predict as pt
import train
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import glob, os
import json
from collections import OrderedDict
import orderedDict

data_dir = 'vehicles'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.RzandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.RandomCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
train_data = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32) 
testloader = torch.utils.data.DataLoader(test_data, batch_size=32) 


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        return F.log_softmax(x, dim=1)
vgg16 = models.vgg16()
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images.resize_(images.shape[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
epochs = 2
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    model.train()
    for images, labels in trainloader:
        steps += 1
        
        images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            
            model.eval()
            
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            model.train()

            
save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
model.class_to_idx = image_datasets['train'].class_to_idx
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


checkpoint = torch.load('flowers')
model = getattr(torchvision.models, checkpoint['arch'])
model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['state_dict'])
model.optimizer = checkpoint('optimizer')
return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    if img.size[0]>=img.size[1]: 
        img.thumbnail((10000,256)) 
    else: 
        img.thumbnail((256,10000))    half_the_width = image.size[0] / 2
    half_the_height = image.size[1] / 2
    image = image.crop((half_the_width - 112,
                       half_the_height - 112,
                       half_the_width + 112,
                       half_the_height + 112))

    np_image = np.array(image)
    np_image = np.array(np_image)/255

    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])

    image = (np_image - mean) / std_dev
    image = image.transpose((2, 0, 1))

    return torch.from_numpy(image)



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)  
    return ax


def predict('flowers', model, topk=5):
    probs, classes = predict('flowers', model)
    probs = class_to_idx
    k = torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
    torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
    return probs
    return classes


k.imshow(topk)
print(torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor))

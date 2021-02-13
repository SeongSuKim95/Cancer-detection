import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image
from torchvision.transforms.transforms import ToPILImage
from Dataloader import Breast_cancer_dataset
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs')

# Represent output channels after Conv layer
VGG_types = {
    'VGG_SS': [64, 128, 'M', 256, 256, 'M', 512, 512 , 'M', 512, 512], 
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512 , 512, 'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

# Then Flatten 4096 * 4096 * 1000 Linear layers

class VGG_net(nn.Module):

    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG_net, self).__init__()

        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG_types['VGG_SS'])
        self.fcs = nn.Sequential(
                    nn.Linear(512*7*7,4096),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096,4096),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, num_classes)
        )

    def forward(self,x):

        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        
        x = self.fcs(x)

        return x

    def create_conv_layer(self,architecture):

        layers = []

        in_channels = self.in_channels
         
        ## Simliar method to ResNet
        for x in architecture:

            if type(x) == int: # Conv layers
                out_channels = x
                layers += [nn.Conv2d(in_channels = in_channels,out_channels = out_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU(),]
                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
            
        return nn.Sequential(*layers)


def save_checkpoint(state,filename):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VGG_net(in_channels = 3, num_classes = 2).to(device)
# x = torch.randn(16,3,224,224).to(device)
# print(model(x))

# Hyperparams

num_epochs = 50
batch_size = 64
learning_rate = 0.001
load_model = False

# Dataset Transform
transform = transforms.Compose([    
                                transforms.ToPILImage(),
                                transforms.Resize((56,56), interpolation = 2),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5,0.5,0.5])
                                ])

train_df = pd.read_csv("./train_csv.csv")
print(len(train_df))
print(train_df.groupby("label")["label"].count())

train,test = train_test_split(train_df, test_size =0.2, stratify = train_df.label)

train_dataset = Breast_cancer_dataset(df_data = train, transform = transform)
test_dataset = Breast_cancer_dataset(df_data = test, transform = transform)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 128, shuffle = True, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 128, shuffle = False, num_workers = 0)

# For Sanity check
# images, labels = next(iter(train_loader))

# print(total_step)
total_step = len(train_loader)

# Criterion
criterion = nn.CrossEntropyLoss()

# Optimizer

optimizer = optim.SGD(model.parameters(),lr = learning_rate, momentum = 0.9, weight_decay =  0.0002)

# Load

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Train the model
model.train()
running_loss = 0
running_correct = 0
for epoch in range(num_epochs):
        #if epoch % 20 == 0 : --> for sanity check
        #print(f"Epoch[{epoch+1}/{num_epochs}]")
        losses = []
        if epoch % 5 == 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
            filename = "./save/checkpoint_%d.pth.tar" % (epoch)
            save_checkpoint(checkpoint,filename)

        for i, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss  = criterion(outputs,labels)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            running_correct += (predicted == labels).sum().item()

            ## check loss, steps per each 100 steps
            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss : {:.4f}".format(epoch+1,num_epochs,i+1,total_step,loss.item()))
                ############# TENSORBOARD ########################
                #Loss(Scalar) -> writer.add_scalar
                writer.add_scalar('training loss', running_loss / 100, epoch * total_step + i)
                running_accuracy = running_correct / predicted.size(0) / 100
                #predicted.size(0) = batch size
                #predicted.size(0) * 100 ->>> Num of step in current epoch
                writer.add_scalar('accuracy', running_accuracy, epoch * total_step + i)

                ##Clear for accuracy, loss
                running_correct = 0
                running_loss = 0.0
                ###################################################


model.eval()
with torch.no_grad():

    correct = 0 
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _ , predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    
    print('Accuaracy of the model on the test images: {}%'.format(100*correct/total))

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import PIL.Image as Image
from torchvision.transforms.transforms import ToPILImage
from Dataloader import Breast_cancer_dataset
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/20epoch_BCE_Adam')
os.environ["CUDA_VISIBLE_DEVICES"]="1"

##Creating Basic Block

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample = None, stride = 1):
         super(BasicBlock,self).__init__()

         self.expansion = 4
        
         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
         self.bn1 = nn.BatchNorm2d(out_channels)
         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1) # Maintain size
         self.bn2 = nn.BatchNorm2d(out_channels)
         self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0)
         self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
         self.relu = nn.ReLU()
   
         self.identity_downsample = identity_downsample
          
         # Ex) [1x1 64 , 3x3 64, 1x1 256] x 3
         # conv1 = 1x1, conv2 = 3x3, conv3 = 1x1
         # identity_downsample = skip_connection

    def forward(self,x):

        # Residual block
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:

            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x


class ResNet(nn.Module): #[3, 4, 6, 3]
    def __init__(self,block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64 # Initial Resnet block Input in_channels
        #self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding =3)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, padding =3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        #self.maxpool = nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        # ResNet layers
        self.layer1 = self._make_layers(block,layers[0],out_channels=64,stride = 1) # 64 64 256 Stride = 1
        self.layer2 = self._make_layers(block,layers[1],out_channels=128,stride = 2) # 128 128 512 Stride = 2
        self.layer3 = self._make_layers(block,layers[2],out_channels=256,stride = 2) # 256 256 1024 Stride = 2 
        self.layer4 = self._make_layers(block,layers[3],out_channels=512,stride = 2) # 512 512 2048 Stride = 2

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)

    def forward(self,x):
        x = self.conv1(x) # stride = 2
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x) # kernel_size = 2

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)

        x = self.fc(x)

        return x

    def _make_layers(self,block,num_residual_blocks,out_channels,stride): # block = Basicblock , num_residual_blocks = [3,4,6,3]

        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        # Changing number of channels
        layers.append(BasicBlock(self.in_channels, out_channels, identity_downsample, stride)) #out_channels = 64
        self.in_channels = out_channels*4 # 256

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels,out_channels)) # 256 -> 64 , 64 *4 (256) again

        return nn.Sequential(*layers)


def ResNet50(img_channels = 3, num_classes = 1000):
    return ResNet(BasicBlock,[3,4,6,3],img_channels,num_classes)

def ResNet101(img_channels = 3, num_classes = 1000):
    return ResNet(BasicBlock,[3,4,23,3],img_channels,num_classes)

def ResNet152(img_channels = 3, num_classes = 1000):
    return ResNet(BasicBlock,[3,8,36,3],img_channels,num_classes)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(state,filename):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet50(img_channels = 3, num_classes = 2).to(device)

# Hyperparams

num_epochs = 30
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
criterion = nn.BCEWithLogitsLoss()

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
        if (epoch + 1 )% 5 == 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
            filename = "./save/20epoch_BCE_Adam/checkpoint_%d.pth.tar" % (epoch)
            save_checkpoint(checkpoint,filename)

        for i, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss  = criterion(outputs.flatten(),labels.float())
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            running_correct += (predicted == labels).sum().item()
            ## check loss, steps per each 100 steps
            if (i+1) % 100 == 0:
                print(running_correct)
                print("Epoch [{}/{}], Step [{}/{}] Loss : {:.4f}".format(epoch+1,num_epochs,i+1,total_step,loss.item()))
                writer.add_scalar('Training loss per 100 iteration', loss.item(), epoch * total_step + i)
        ############# TENSORBOARD ########################
        #Loss(Scalar) -> writer.add_scalar
        writer.add_scalar('Training loss per epoch', running_loss / total_step, epoch)
        running_accuracy = (running_correct / len(train_dataset)) * 100
        #predicted.size(0) = batch size
        #predicted.size(0) * 100 ->>> Num of step in current epoch
        writer.add_scalar('Train Accuracy per epoch', running_accuracy, epoch)
        print("Epoch [{}] Training loss = {}, Train Accuracy = {} % ".format(epoch, running_loss / total_step, running_accuracy))
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

            writer.add_scalar('Test Accuracy per epoch',  100 * correct/total, epoch )
            print('Accuaracy of the model on the test images: {}%'.format(100*correct/total))
        model.train()
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import os
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image
from torchvision.transforms.transforms import ToPILImage
from Dataloader import Breast_cancer_dataset
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/20epoch_CE_Adam')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
class GoogleNet(nn.Module):

    def __init__(self,in_channels = 3, num_classes = 2):
        super(GoogleNet,self).__init__()

        self.conv1 = conv_block(in_channels = in_channels, out_channels = 64 , kernel_size = (7,7), stride = (2,2), padding = (3,3))

        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = conv_block(in_channels = 64, out_channels = 192, kernel_size = 3 , stride = 1, padding =1 )
        self.maxpool2 = nn.MaxPool2d(kernel_size = (3,3), stride = 2, padding = 1)

        # In this order --> in_channels, out_1x1, reduction_3x3, out_3x3, reduction_5x5, out_5x5, out_1x1pool

        self.inception3a = Inception_block(in_channels = 192, out_1x1 = 64, reduction_3x3 = 96, out_3x3 = 128, reduction_5x5 = 16, out_5x5 = 32, out_1x1pool = 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride =2, padding =1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 3, stride =2 , padding =1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024,2)
    
    def forward(self,x):

        x = self.conv1(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        #x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        ##Important
        x = x.reshape(x.shape[0],-1)
        ##
        x = self.dropout(x)
        x = self.fc1(x)

        return x
        
class Inception_block(nn.Module):

    def __init__(self, in_channels, out_1x1, reduction_3x3, out_3x3, reduction_5x5, out_5x5, out_1x1pool):
        super(Inception_block,self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size = (1,1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, reduction_3x3, kernel_size = (1,1)),
            conv_block(reduction_3x3, out_3x3, kernel_size = (3,3), stride = 1, padding = (1,1))
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, reduction_5x5, kernel_size = (1,1)),
            conv_block(reduction_5x5, out_5x5, kernel_size = (5,5), stride = 1, padding = (2,2))
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            conv_block(in_channels, out_1x1pool, kernel_size = (1,1))
        )
    
    def forward(self,x):

        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],1)

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels, **kwargs):
        super(conv_block,self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) #kernel_size = (1,1), (3,3), (5,5)
        self.bn = nn.BatchNorm2d(out_channels)    

    def forward(self,x):

        return self.relu(self.bn(self.conv(x)))

def save_checkpoint(state,filename):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GoogleNet(in_channels = 3, num_classes = 1).to(device)

num_epochs = 20
batch_size = 64
learning_rate = 3e-4
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

optimizer = optim.Adam(model.parameters(),lr = learning_rate)

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
            filename = "./save/20epoch_CE_Adam/checkpoint_%d.pth.tar" % (epoch)
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

            writer.add_scalar('Test Accuracy per epoch',  100 * correct/total, epoch + 1 )
            print('Accuaracy of the model on the test images: {}%'.format(100*correct/total))
        model.train()


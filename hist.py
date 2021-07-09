import os
import time
# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import glob
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import shutil
# torch.manual_seed(1)    # reproducible


             
BATCH_SIZE = 20
LR = 0.001              
EPOCH = 1



#################################  FIXED BELOW ################################# 
#####  at most excute once  #####
os.mkdir('/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-TEST')
os.mkdir('/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-TEST/no')
os.mkdir('/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-TEST/yes')
#####  at most excute once  #####


CLOUD_PATH = '/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-DATA/cloudstreet'
CLOUD_TEST_PATH = '/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-TEST'
CLOUD_PATH1 = '/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-DATA/cloudstreet/yes'
CLOUD_PATH2 = '/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-DATA/cloudstreet/no'
CLOUD_TEST_PATH1='/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-TEST/yes'
CLOUD_TEST_PATH2='/Users/gumenghan/Desktop/NASA-IMPACT-CLOUD-TEST/no'
Paths1 = glob.glob(CLOUD_PATH1+'/*.jpg')

Paths2 = glob.glob(CLOUD_PATH2+'/*.jpg')
test_P1 =glob.glob(CLOUD_TEST_PATH1+'/*.jpg')
test_P2 =glob.glob(CLOUD_TEST_PATH2+'/*.jpg')


JPG1,JPG2=[],[]
for item in Paths1:
    tmp=item.split('/')
    JPG1.append(tmp[-1])
    
for item in Paths2:
    tmp=item.split('/')
    JPG2.append(tmp[-1])

#  Move the last 10% JPGs to new folder  #
spli1,spli2=int(9*len(JPG1)/10),int(9*len(JPG2)/10)

test_p1, test_p2 = JPG1[spli1:],JPG2[spli2:]

#################################
#####  at most excute once  #####
for file in test_p1:
    shutil.move(CLOUD_PATH1+'/'+file,CLOUD_TEST_PATH1+'/'+file)
for file in test_p2:
    shutil.move(CLOUD_PATH2+'/'+file,CLOUD_TEST_PATH2+'/'+file)
#####  at most excute once  #####
#################################
    
#tag=[]
#for item in paths2:
#    tag.append(0)
#for item in paths1:
#    tag.append(1)

#target = torch.FloatTensor(tag)
#################################  FIXED ABOVE #################################

normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    #transforms.RandomVerticalFlip(), 
    transforms.ToTensor(), 
    normalize])


train_dataset = ImageFolder(CLOUD_PATH, transform=train_transform)

#print(len(train_dataset.imgs))
#train_dataset.targets=target
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


test_dataset = ImageFolder(CLOUD_TEST_PATH, transform=train_transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 100, 100)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,
                kernel_size=5,              
                stride=1,                   
                padding=2,                  # if same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 100, 100)
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    # max value in 2x2 area, output shape (16, 50, 50)
        )
        self.conv2 = nn.Sequential(         #  (16, 50, 50)
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                # (32, 25, 25)
        )
        self.fc = nn.Linear(32 * 32 * 32, 120)
        self.out = nn.Linear(120, 2)
        #self.out = nn.Linear(32 * 25 * 25, 2)   # fully connected layer, output 2 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = F.relu(self.fc(x))
        output = self.out(x)
        return output    # return x for visualization


cnn = CNN()


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   
loss_func = nn.CrossEntropyLoss()                      

#loss_stor,step_stor=[],[]

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_dataloader):   
        
        output = cnn(b_x)             
        loss = loss_func(output, b_y)
        
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()         
        
        if step % 10 == 0:
            print(loss)
        #    loss_stor.append(loss)
        #    step_stor.append(step)

correct1,correct2=0,0
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(test_dataloader):   
        
        output = cnn(b_x)   
        output.tolist()
        
        for i in range(0,len(output)): 
            res=0
            if output[i][0]<=output[i][1]:
                res=1
            #print(output[i][0], output[i][1],b_y[i])
            if res==b_y[i]:
                if res==0:
                    correct1=correct1+1
                else:
                    correct2=correct2+1

print(correct1, correct2)
true_positive=correct2
true_negative=correct1

false_negative=EPOCH*len(test_p1)-true_positive
false_positive=EPOCH*len(test_p2)-true_negative


Accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
Precision = true_positive / (true_positive + false_positive) ##Precision##


Recall = true_positive / (true_positive + false_negative) ##Recall##

F1_score = 2* (Precision*Recall)/(Precision+Recall)
print('Accuracy is:', Accuracy)
print('Precision is:', Precision)



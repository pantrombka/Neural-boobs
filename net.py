import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

data = ImageFolder(root='img-train', transform= transforms.ToTensor())
data2 = ImageFolder(root='img-valid', transform= transforms.ToTensor())
trainloader = DataLoader(data)
testloader = DataLoader(data2)
classes = ('before', 'after')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 320, kernel_size=5)
        self.conv2 = nn.Conv2d(320, 640, kernel_size=5)
        self.conv3 = nn.Conv2d(640, 1280, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(80640, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 2)

    def forward(self, x):    
        x = F.relu(F.max_pool2d(self.conv1(x), 2))          
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv3(x)), 2))
        x = x.view(-1, 80640 )    
        x = F.relu(self.fc1(x))	
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # 50 -> 10
        x = self.fc3(x)       
        # transform to logits
        return F.log_softmax(x)
   
net=Net()
net.cuda() 
print(net)
import torch.optim as optim

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.0022)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.20f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)
dataiter = iter(testloader)
images, labels = dataiter.next()


net.eval()

correct = 0
total = 0

for data in testloader:
	images, labels = data
	images=images.cuda()
	labels=labels.cuda()
	outputs = net(Variable(images))
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()
    
print('Correct: %d %%' % (
    100 * correct / total))
print(correct)
print(total)


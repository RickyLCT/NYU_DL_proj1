import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available')
else:
    print('CUDA is available')
     

# set the hyperparameter
batch_size = 64
learning_rate = 0.01
epoch_num = 100

# load the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
#transform = transforms.Compose([transforms.ToTensor()])
                                
traindata = datasets.CIFAR10('data', train=True, download=True, transform=transform)
traindata, validata = torch.utils.data.random_split(traindata,[40000, 10000])
testdata = datasets.CIFAR10('data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
vali_loader = torch.utils.data.DataLoader(validata, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [2, 2, 2, 2])



net = project1_model().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)

train_loss_history = []
vali_loss_history = []
train_acc_history = []
vali_acc_history = []

def main():
    for epoch in range(epoch_num):
        
        train_loss = 0.0
        train_acc = 0.0
        net.train()
        for i, train_data in enumerate(train_loader):
            inputs, labels = train_data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            # forward + backward
            predicted_output = net(inputs)
            fit = loss(predicted_output, labels)
            fit.backward()
            optimizer.step()
            train_loss += fit.item()
            _, pred = predicted_output.max(1)
            num_correct = (pred==labels).sum().item()
            acc = num_correct / inputs.shape[0]
            train_acc += acc
        train_loss = train_loss/len(train_loader)
        train_loss_history.append(train_loss)
        train_acc = train_acc / len(train_loader)
        train_acc_history.append(train_acc)
        
        vali_loss = 0.0
        vali_acc = 0.0
        net.eval()
        for j, data in enumerate(vali_loader):
            with torch.no_grad():
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                predicted_output = net(images)
                fit = loss(predicted_output,labels)
                vali_loss += fit.item()
                _, pred = predicted_output.max(1)
                num_correct = (pred==labels).sum().item()
                acc = num_correct / images.shape[0]
                vali_acc += acc
        vali_loss = vali_loss / len(vali_loader)
        vali_loss_history.append(vali_loss)
        vali_acc = vali_acc / len(vali_loader)
        vali_acc_history.append(vali_acc)
        print('Epoch %s, Train loss %.6f, Validation loss %.6f, Train acc %.6f, Validation acc %.6f'%(epoch, train_loss, vali_loss, train_acc, vali_acc))
        
    
    torch.save({'model':net.state_dict()}, './model_file/project1_model.pt')
     
    plt.plot(range(epoch_num),train_loss_history,'-',linewidth=3,label='Train error')
    plt.plot(range(epoch_num),vali_loss_history,'-',linewidth=3,label='Validation error')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()


    plt.plot(range(epoch_num),train_acc_history,'-',linewidth=3,label='Train accuracy')
    plt.plot(range(epoch_num),vali_acc_history,'-',linewidth=3,label='Validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend()
    
    test_loss = 0.0
    num_correct = 0
    for j, data in enumerate(test_loader):
        with torch.no_grad():
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            predicted_output = net(images)
            fit = loss(predicted_output,labels)
            test_loss += fit.item()
            _, pred = predicted_output.max(1)
            num_correct += (pred==labels).sum().item()
    test_loss = test_loss / len(test_loader)
    test_acc = num_correct / len(testdata)
    print(num_correct)
    print('Testing set, Average loss %.6f, Average acc = {%.0f} / {%.0f} = %.6f'%(test_loss, num_correct, len(testdata), test_acc))
    
if __name__ == '__main__':
    main()
            
            
    




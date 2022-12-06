from pickle import TRUE
from sympy import true
import torch
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import datasets
from torchvision import transforms
from alxnet import VGGNet16, alxnet
from  torch import device, logit, nn,optim
import torchvision
from torch.nn import functional as F
import numpy as np




batchsz=16
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('alxnet/cifar', train=True, download=True, #加载数据集，下载 teain是否作为训练   download 如果数据空自动下载
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),     
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size=batchsz, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('alxnet/cifar/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size=1, shuffle=True)



device=torch.device('cuda')
model=VGGNet16().to(device)
#model.load_state_dict(torch.load('encdo_11.pth'))
criteon=nn.CrossEntropyLoss().to(device)
optimizeer=optim.Adam(model.parameters(),lr=0.001)
test_loss=0.0


for epoch in range(4):
    model.train()
    ciloss=0.0
    m=0  
    train_loss = 0.0 
    for data, target in train_loader:
        m+=1
        #if m==750:break
        data=data.to(device)
        target=target.to(device)
        optimizeer.zero_grad()
        output = model(data)
        loss = criteon(output, target)
        loss.backward()
        optimizeer.step()
        train_loss += loss.item()

    train_loss = train_loss / (len(train_loader)*0.5)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))





class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
PATH='do2model11.pth'
    # model in evaluation mode
model.eval()




for data, target in test_loader:
        data=data.to(device)
        target=target.to(device)
        output = model(data)
        loss = criteon(output, target)
        
        loss = criteon(output, target)
        test_loss += loss.item()
        
        # 将输出概率转为lable
        _, pred = torch.max(output, 1)
        # 比较
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # 计算每个类的准确个数
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
    
torch.save(model.state_dict(),PATH)

    # 打印平均loss
test_loss = test_loss / sum(class_total)
print(f'Test Loss: {test_loss:.6f}\n')

for label in range(10):
    print(
            f'Test Accuracy of {label}: {float(100 * class_correct[label] / class_total[label])}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

print(
    f'\nTest Accuracy (Overall): {float(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
    f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )
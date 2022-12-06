from pickle import FALSE
from statistics import mode
from sympy import true
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from alxnet import VGGNet16, alxnet
from  torch import device, logit, nn,optim
import torchvision
from torch.nn import functional as F
import numpy as np
import tenseal as ts
import os
import time

start=time.time()
coeff_mod_bit_sizes = [20,14,20]
# create TenSEALContext
ctx = ts.context(ts.SCHEME_TYPE.CKKS, 2048, -1, coeff_mod_bit_sizes)
ctx.global_scale = 2 ** 15
ctx.generate_galois_keys()
end=time.time()


m1conv0=torch.randn(16,3,3,3)
m1conv2=torch.randn(32,16,3,3)
m1conv5=torch.randn(64,32,3,3)
m1conv7=torch.randn(64,64,3,3)
m1conv10=torch.randn(128,64,3,3)
m1conv12=torch.randn(128,128,3,3)
m1conv14=torch.randn(256,128,3,3)
m1conv17=torch.randn(256,256,3,3)
m1fc0=torch.randn(256,4096)
m1fc3=torch.randn(128, 256)
m1fc6=torch.randn(10, 128)
print('密钥生成时间;',float(end-start))

#part1
m=1
model=VGGNet16()
model.load_state_dict(torch.load('D:\python\\do1model11.pth'))

model_dict=model.state_dict()
"""print(model_dict.keys())

for name in model_dict.keys():

   print(model_dict[name].shape)"""
start1=time.time()
mm1conv0=model_dict['Conv1.0.weight']+m1conv0

mm1conv2=model_dict['Conv1.2.weight']+m1conv2

mm1conv5=model_dict['Conv2.0.weight']+m1conv5

mm1conv7=model_dict['Conv2.3.weight']+m1conv7

mm1conv10=model_dict['Conv3.0.weight']+m1conv10

mm1conv12=model_dict['Conv3.3.weight']+m1conv12

mm1conv14=model_dict['Conv3.6.weight']+m1conv14

mm1conv17=model_dict['Conv3.8.weight']+m1conv17
"""m1conv19=torch.randn(512,512,3,3)
mm1conv19=model_dict['Conv4.3.weight']+m1conv19
m1conv21=torch.randn(512,512,3,3)
mm1conv21=model_dict['Conv4.6.weight']+m1conv21
m1conv24=torch.randn(512,512,3,3)
mm1conv24=model_dict['Conv5.0.weight']+m1conv24
m1conv26=torch.randn(512,512,3,3)
mm1conv26=model_dict['Conv5.3.weight']+m1conv26
m1conv28=torch.randn(512,512,3,3)
mm1conv28=model_dict['Conv5.6.weight']+m1conv28"""

mm1fc0=model_dict['fc.0.weight']+m1fc0

mm1fc3=model_dict['fc.3.weight']+m1fc3

mm1fc6=model_dict['fc.6.weight']+m1fc6

mm1convb0=ts.ckks_tensor(ctx,model_dict['Conv1.0.bias'])
mm1convb2=ts.ckks_tensor(ctx,model_dict['Conv1.2.bias'])
mm1convb5=ts.ckks_tensor(ctx,model_dict['Conv2.0.bias'])
mm1convb7=ts.ckks_tensor(ctx,model_dict['Conv2.3.bias'])
mm1convb10=ts.ckks_tensor(ctx,model_dict['Conv3.0.bias'])
mm1convb12=ts.ckks_tensor(ctx,model_dict['Conv3.3.bias'])
mm1convb14=ts.ckks_tensor(ctx,model_dict['Conv3.6.bias'])
mm1convb17=ts.ckks_tensor(ctx,model_dict['Conv3.8.bias'])
"""mm1convb19=ts.ckks_tensor(ctx,model_dict['Conv4.3.bias'])
mm1convb21=ts.ckks_tensor(ctx,model_dict['Conv4.6.bias'])
mm1convb24=ts.ckks_tensor(ctx,model_dict['Conv5.0.bias'])
mm1convb26=ts.ckks_tensor(ctx,model_dict['Conv5.3.bias'])
mm1convb28=ts.ckks_tensor(ctx,model_dict['Conv5.6.bias'])"""
mm1fcb0=ts.ckks_tensor(ctx,model_dict['fc.0.bias'])
mm1fcb3=ts.ckks_tensor(ctx,model_dict['fc.3.bias'])
mm1fcb6=ts.ckks_tensor(ctx,model_dict['fc.6.bias'])
end1=time.time()
print('加密时间：',float(end1-start1))



print(m)
#part2
model1=VGGNet16()
model1.load_state_dict(torch.load('D:\python\\do2model11.pth'))
model_dict1=model1.state_dict()
dict=model_dict
m2conv0=torch.randn(16,3,3,3)
mm2conv0=model_dict1['Conv1.0.weight']+m2conv0
m2conv2=torch.randn(32,16,3,3)
mm2conv2=model_dict1['Conv1.2.weight']+m2conv2
m2conv5=torch.randn(64,32,3,3)
mm2conv5=model_dict1['Conv2.0.weight']+m2conv5
m2conv7=torch.randn(64,64,3,3)
mm2conv7=model_dict1['Conv2.3.weight']+m2conv7
m2conv10=torch.randn(128,64,3,3)
mm2conv10=model_dict1['Conv3.0.weight']+m2conv10
m2conv12=torch.randn(128,128,3,3)
mm2conv12=model_dict1['Conv3.3.weight']+m2conv12
m2conv14=torch.randn(256,128,3,3)
mm2conv14=model_dict1['Conv3.6.weight']+m2conv14
m2conv17=torch.randn(256,256,3,3)
mm2conv17=model_dict1['Conv3.8.weight']+m2conv17
"""m2conv19=torch.randn(512,512,3,3)
mm2conv19=model_dict['Conv4.3.weight']+m2conv19
m2conv21=torch.randn(512,512,3,3)
mm2conv21=model_dict['Conv4.6.weight']+m2conv21
m2conv24=torch.randn(512,512,3,3)
mm2conv24=model_dict['Conv5.0.weight']+m2conv24
m2conv26=torch.randn(512,512,3,3)
mm2conv26=model_dict['Conv5.3.weight']+m2conv26
m2conv28=torch.randn(512,512,3,3)
mm2conv28=model_dict['Conv5.6.weight']+m2conv28"""
m2fc0=torch.randn(256,4096)
mm2fc0=model_dict1['fc.0.weight']+m2fc0
m2fc3=torch.randn(128, 256)
mm2fc3=model_dict1['fc.3.weight']+m2fc3
m2fc6=torch.randn(10, 128)
mm2fc6=model_dict1['fc.6.weight']+m2fc6

mm2convb0=ts.ckks_tensor(ctx,model_dict1['Conv1.0.bias'])
mm2convb2=ts.ckks_tensor(ctx,model_dict1['Conv1.2.bias'])
mm2convb5=ts.ckks_tensor(ctx,model_dict1['Conv2.0.bias'])
mm2convb7=ts.ckks_tensor(ctx,model_dict1['Conv2.3.bias'])
mm2convb10=ts.ckks_tensor(ctx,model_dict1['Conv3.0.bias'])
mm2convb12=ts.ckks_tensor(ctx,model_dict1['Conv3.3.bias'])
mm2convb14=ts.ckks_tensor(ctx,model_dict1['Conv3.6.bias'])
mm2convb17=ts.ckks_tensor(ctx,model_dict1['Conv3.8.bias'])
"""mm2convb19=ts.ckks_tensor(ctx,model_dict['Conv4.3.bias'])
mm2convb21=ts.ckks_tensor(ctx,model_dict['Conv4.6.bias'])
mm2convb24=ts.ckks_tensor(ctx,model_dict['Conv5.0.bias'])
mm2convb26=ts.ckks_tensor(ctx,model_dict['Conv5.3.bias'])
mm2convb28=ts.ckks_tensor(ctx,model_dict['Conv5.6.bias'])"""
mm2fcb0=ts.ckks_tensor(ctx,model_dict1['fc.0.bias'])
mm2fcb3=ts.ckks_tensor(ctx,model_dict1['fc.3.bias'])
mm2fcb6=ts.ckks_tensor(ctx,model_dict1['fc.6.bias'])

#part3
print(m)
model2=VGGNet16()
model2.load_state_dict(torch.load('D:\python\\do3model11.pth'))
model_dict2=model2.state_dict()

m3conv0=torch.randn(16,3,3,3)
mm3conv0=model_dict2['Conv1.0.weight']+m3conv0
m3conv2=torch.randn(32,16,3,3)
mm3conv2=model_dict2['Conv1.2.weight']+m3conv2
m3conv5=torch.randn(64,32,3,3)
mm3conv5=model_dict2['Conv2.0.weight']+m3conv5
m3conv7=torch.randn(64,64,3,3)
mm3conv7=model_dict2['Conv2.3.weight']+m3conv7
m3conv10=torch.randn(128,64,3,3)
mm3conv10=model_dict2['Conv3.0.weight']+m3conv10
m3conv12=torch.randn(128,128,3,3)
mm3conv12=model_dict2['Conv3.3.weight']+m3conv12
m3conv14=torch.randn(256,128,3,3)
mm3conv14=model_dict2['Conv3.6.weight']+m3conv14
m3conv17=torch.randn(256,256,3,3)
mm3conv17=model_dict2['Conv3.8.weight']+m3conv17
"""m3conv19=torch.randn(512,512,3,3)
mm3conv19=model_dict['Conv4.3.weight']+m3conv19
m3conv21=torch.randn(512,512,3,3)
mm3conv21=model_dict['Conv4.6.weight']+m3conv21
m3conv24=torch.randn(512,512,3,3)
mm3conv24=model_dict['Conv5.0.weight']+m3conv24
m3conv26=torch.randn(512,512,3,3)
mm3conv26=model_dict['Conv5.3.weight']+m3conv26
m3conv28=torch.randn(512,512,3,3)
mm3conv28=model_dict['Conv5.6.weight']+m3conv28"""
m3fc0=torch.randn(256,4096)
mm3fc0=model_dict2['fc.0.weight']+m3fc0
m3fc3=torch.randn(128, 256)
mm3fc3=model_dict2['fc.3.weight']+m3fc3
m3fc6=torch.randn(10, 128)
mm3fc6=model_dict2['fc.6.weight']+m3fc6

mm3convb0=ts.ckks_tensor(ctx,model_dict2['Conv1.0.bias'])
mm3convb2=ts.ckks_tensor(ctx,model_dict2['Conv1.2.bias'])
mm3convb5=ts.ckks_tensor(ctx,model_dict2['Conv2.0.bias'])
mm3convb7=ts.ckks_tensor(ctx,model_dict2['Conv2.3.bias'])
mm3convb10=ts.ckks_tensor(ctx,model_dict2['Conv3.0.bias'])
mm3convb12=ts.ckks_tensor(ctx,model_dict2['Conv3.3.bias'])
mm3convb14=ts.ckks_tensor(ctx,model_dict2['Conv3.6.bias'])
mm3convb17=ts.ckks_tensor(ctx,model_dict2['Conv3.8.bias'])
"""mm3convb19=ts.ckks_tensor(ctx,model_dict['Conv4.3.bias'])
mm3convb21=ts.ckks_tensor(ctx,model_dict['Conv4.6.bias'])
mm3convb24=ts.ckks_tensor(ctx,model_dict['Conv5.0.bias'])
mm3convb26=ts.ckks_tensor(ctx,model_dict['Conv5.3.bias'])
mm3convb28=ts.ckks_tensor(ctx,model_dict['Conv5.6.bias'])"""
mm3fcb0=ts.ckks_tensor(ctx,model_dict2['fc.0.bias'])
mm3fcb3=ts.ckks_tensor(ctx,model_dict2['fc.3.bias'])
mm3fcb6=ts.ckks_tensor(ctx,model_dict2['fc.6.bias'])

#聚合
start2=time.time()
print(m)
cov0w=0.333*mm1conv0+0.33*mm2conv0+0.333*mm3conv0
cov2w=0.333*mm1conv2+0.333*mm2conv2+0.333*mm3conv2
cov5w=0.333*mm1conv5+0.333*mm2conv5+0.333*mm3conv5
cov7w=0.333*mm1conv7+0.333*mm2conv7+0.333*mm3conv7
cov10w=0.333*mm1conv10+0.333*mm2conv10+0.333*mm3conv10
cov12w=0.333*mm1conv12+0.333*mm2conv12+0.333*mm3conv12
cov14w=0.333*mm1conv14+0.333*mm2conv14+0.333*mm3conv14
cov17w=0.333*mm1conv17+0.333*mm2conv17+0.333*mm3conv17
"""cov19w=0.2*mm1conv19+0.3*mm2conv19+0.5*mm3conv19
cov21w=0.2*mm1conv21+0.3*mm2conv21+0.5*mm3conv21
cov24w=0.2*mm1conv24+0.3*mm2conv24+0.5*mm3conv24
cov26w=0.2*mm1conv26+0.3*mm2conv26+0.5*mm3conv26
cov28w=0.2*mm1conv28+0.3*mm2conv28+0.5*mm3conv28"""
fc0w=0.3333*mm1fc0+0.333*mm2fc0+0.333*mm3fc0
fc3w=0.333*mm1fc3+0.333*mm2fc3+0.333*mm3fc3
fc6w=0.333*mm1fc6+0.333*mm2fc6+0.333*mm3fc6

cov0b=0.333*mm1convb0+0.333*mm2convb0+0.333*mm3convb0
cov2b=0.333*mm1convb2+0.333*mm2convb2+0.333*mm3convb2
cov5b=0.333*mm1convb5+0.333*mm2convb5+0.333*mm3convb5
cov7b=0.333*mm1convb7+0.333*mm2convb7+0.333*mm3convb7
cov10b=0.333*mm1convb10+0.333*mm2convb10+0.333*mm3convb10
cov12b=0.333*mm1convb12+0.333*mm2convb12+0.333*mm3convb12
cov14b=0.333*mm1convb14+0.333*mm2convb14+0.333*mm3convb14
cov17b=0.333*mm1convb17+0.333*mm2convb17+0.333*mm3convb17
"""cov19b=0.2*mm1convb19+0.3*mm2convb19+0.5*mm3convb19
cov21b=0.2*mm1convb21+0.3*mm2convb21+0.5*mm3convb21
cov24b=0.2*mm1convb24+0.3*mm2convb24+0.5*mm3convb24
cov26b=0.2*mm1convb26+0.3*mm2convb26+0.5*mm3convb26
cov28b=0.2*mm1convb28+0.3*mm2convb28+0.5*mm3convb28"""
fc0b=0.333*mm1fcb0+0.333*mm2fcb0+0.333*mm3fcb0
fc3b=0.333*mm1fcb3+0.333*mm2fcb3+0.333*mm3fcb3
fc6b=0.333*mm1fcb6+0.333*mm2fcb6+0.333*mm3fcb6

end2=time.time()
print('聚合时间：',float(end2-start2))




start3=time.time()
#除噪音
dict['Conv1.0.weight']=cov0w-0.333*m1conv0-0.333*m2conv0-0.333*m3conv0
dict['Conv1.2.weight']=cov2w-0.333*m1conv2-0.333*m2conv2-0.333*m3conv2
dict['Conv2.0.weight']=cov5w-0.333*m1conv5-0.333*m2conv5-0.333*m3conv5
dict['Conv2.3.weight']=cov7w-0.333*m1conv7-0.333*m2conv7-0.333*m3conv7
dict['Conv3.0.weight']=cov10w-0.333*m1conv10-0.333*m2conv10-0.333*m3conv10
dict['Conv3.3.weight']=cov12w-0.333*m1conv12-0.333*m2conv12-0.333*m3conv12
dict['Conv3.6.weight']=cov14w-0.333*m1conv14-0.333*m2conv14-0.333*m3conv14
dict['Conv3.8.weight']=cov17w-0.333*m1conv17-0.333*m2conv17-0.333*m3conv17
"""dict['Conv4.3.weight']=cov19w-0.2*m1conv19-0.3*m2conv19-0.5*m3conv19
dict['Conv4.6.weight']=cov21w-0.2*m1conv21-0.3*m2conv21-0.5*m3conv21
dict['Conv5.0.weight']=cov24w-0.2*m1conv24-0.3*m2conv24-0.5*m3conv24
dict['Conv5.3.weight']=cov26w-0.2*m1conv26-0.3*m2conv26-0.5*m3conv26
dict['Conv5.6.weight']=cov28w-0.2*m1conv28-0.3*m2conv28-0.5*m3conv28"""
dict['fc.0.weight']=fc0w-0.333*m1fc0-0.333*m2fc0-0.333*m3fc0
dict['fc.3.weight']=fc3w-0.333*m1fc3-0.333*m2fc3-0.333*m3fc3
dict['fc.6.weight']=fc6w-0.333*m1fc6-0.333*m2fc6-0.333*m3fc6

#解密
dict['Conv1.0.bias']=torch.Tensor(cov0b.decrypt().raw)
dict['Conv1.2.bias']=torch.Tensor(cov2b.decrypt().raw)
dict['Conv2.0.bias']=torch.Tensor(cov5b.decrypt().raw)
dict['Conv2.3.bias']=torch.Tensor(cov7b.decrypt().raw)
dict['Conv3.0.bias']=torch.Tensor(cov10b.decrypt().raw)
dict['Conv3.3.bias']=torch.Tensor(cov12b.decrypt().raw)
dict['Conv3.6.bias']=torch.Tensor(cov14b.decrypt().raw)
dict['Conv3.8.bias']=torch.Tensor(cov17b.decrypt().raw)
"""dict['Conv4.3.bias']=torch.Tensor(cov19b.decrypt().raw)
dict['Conv4.6.bias']=torch.Tensor(cov21b.decrypt().raw)
dict['Conv5.0.bias']=torch.Tensor(cov24b.decrypt().raw)
dict['Conv5.3.bias']=torch.Tensor(cov26b.decrypt().raw)
dict['Conv5.6.bias']=torch.Tensor(cov28b.decrypt().raw)"""
dict['fc.0.bias']=torch.Tensor(fc0b.decrypt().raw)
dict['fc.3.bias']=torch.Tensor(fc3b.decrypt().raw)
dict['fc.6.bias']=torch.Tensor(fc6b.decrypt().raw)

end3=time.time()
print('解密时间:',float(end3-start3))





model.load_state_dict(dict)

torch.save(model.state_dict(),'D:\python\encdo_11.pth')
"""for name in model_dict.keys():
    model_dict3[name]=0.25*model_dict[name]+0.5*model_dict2[name]+model_dict1[name]*0.25"""




#model3=alxnet().to(device)
#model3.load_state_dict(model_dict3)







"""criteon=nn.CrossEntropyLoss().to(device)
optimizeer=optim.Adam(model.parameters(),lr=0.001)
test_loss=0.0


for epoch in range(1):
    model.train()
    ciloss=0.0
    m=0  
    train_loss = 0.0 
    for data, target in train_loader:
        m+=1
        if m==600:break
        data=data.to(device)
        target=target.to(device)
        optimizeer.zero_grad()
        output = model(data)
        loss = criteon(output, target)
        loss.backward()
        optimizeer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))





class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
PATH='do1model.pth'
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
    )"""
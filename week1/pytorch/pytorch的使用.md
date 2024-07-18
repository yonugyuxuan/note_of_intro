# pytorch的使用

# 采用pytorch进行训练的基本使用方法

## 1.数据导入

```python
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size, shuffle=True)#True for trainning,False for testing
fron torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):#从dataset中定义
	#初始化
	def _init_(self, file):
		self.data = 
	#每次返回一个sample，用于分Batch
	def _getiten_(self, index): 
		return self.data[index]
	#定义长度函数
	def _len_(self): 
		return len(self.data)

```

关于测试时的dataloader是否需要从新选取？

## 2.反向传播和梯度输出

```python
x=torch.tensor([[1,0],[-1,0]],requires_grad=True)#定义tensor
z=x.pow(2).sum()#进行运算
z.backward()#反向传播
x.grad#输出梯度
```

## 3.pytorch的单层神经网络

```python
import torch.nn as nn
#线性神经网络
nn.linear(in_features,out_features)
#激活函数
nn.Sigmoid()
nn.ReLU()
```

## 4.定义神经网络模型的一般形式

```python
import torch.nn as nn

#1.模型定义
class Mymodel(nn.Module):#父类是nn.Module
	def __init__(self):
		super(MyModel,self).__init__()#调用父类的初始化函数
		self.net=nn.Sequential(
				nn.linear(10,32),
				nn.Sigmoid(),
				nn.Linear(32,1)
		)#神经网络定义并赋值给self.net

#2.前向传播定义
	def forward(self,x):
		return self.net(x)
```

## 5.Loss Function的定义

```python
creiterion=nn.MSELoss()#nn.CrossEntropyLoss()#损失函数定义
loss=creterion(model_output,expected_value)#进行计算
```

## 6.优化方法的定义

```python
torch.option.SGD(model.patameters(),lr,momentum)

#优化的流程
optimizer.zero_grad()#梯度归零
loss.backward()#反向传播求梯度 
optimizer.step()#参数优化
```

# 具体流程

![Untitled](pytorch%E7%9A%84%E4%BD%BF%E7%94%A8%20ba4dac6ad3ef4735ac0ffef2cc0ee7f2/Untitled.png)

## 1.模型初始化

```python
dataset = MyDataset(file)
tr_set = DataLoader(dataset, 16, shuffle=True)#数据
model = MyModel().to(device)#模型
criterion = nn.MSELoss()#损失函数
optimizer = torch.optin.SCD(model.parameters(), 0.1)#优化方法
```

## 2.模型training

```python
for epoch in range(n_epochs):
	model.train() #设置模型模式
	for x, y in tr_set:
		optimizer.zero_grad() #1.消除梯度
		x, y = x.to(device),y.to(device) #2.数据迁移
		pred =model(x)#3.计算
		loss = criterion(pred, y)
		loss.backward()#4.反向传播
		optimizer.step() #5.优化
```

## 3.模型validation

```python
model.eval()#设置模型模式
total_loss =0
for x, y in dv_set:
	x, y = x.to(device), y.to(device)#数据迁移
	with torch.no_grad():#关闭梯度计算
		pred = model(x)
		loss = criterion(pred, y)#计算Loss
	total_loss += loss.cpu().iten()* len(x)
	avg_loss = total_loss / len(dv_set.dataset)
```

## 4.模型test

```python
#基本同validation
model.eval()
preds=[]
for x in tt_set:
	x = x.to(device)
	with torch.no_grad():
		pred = nodel(x)
		preds.append(pred.cpu())
```

## 模型设置→进入循环→数据迁移→计算

[pytorch常用功能](https://www.notion.so/pytorch-fd35494211504a94b59dc79e384fdfac?pvs=21)
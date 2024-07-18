# pytorch常用功能

```python
#生成tensor
torch.tensor(list)
torch.from_numpy(array)
torch.zeros()/ones()
```

```python
#tensor维度相关
torch.squeese(dimension)
torch.unsqueese(dimension)
torch.transpose(dim1,dim2)#转置
torch.cat([tensor1,tensor2,..],dimension)
```

```python
#设备相关
t1.to("cuda")
t1.to("cpu")
torch.cuda.avalible()
```

```python
#查看层形状
layer.weight.shape
layer.bias.shape

```
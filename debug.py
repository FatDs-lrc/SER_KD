import torch
import torch.nn as nn
from torchvision.models import resnet18

class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1, bias=False)
        self.linear = nn.Linear(32*10*10, 20, bias=False)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear(x.view(x.size(0), -1))
        return x
 
model = Simple()
for m in model.parameters():
    m.data.fill_(0.1)
 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
 
model.train()
images = torch.ones(8, 3, 10, 10)
targets = torch.ones(8, dtype=torch.long)
 
output = model(images)
 
loss = criterion(output, targets)
 
print(model.conv1.weight.grad)
# None
loss.backward()
print(model.conv1.weight.grad.data)
 
# print(model.conv1.weight[0][0][0])
 
# optimizer.step()
# print(model.conv1.weight[0][0][0])
 
# optimizer.zero_grad()
# print(model.conv1.weight.grad[0][0][0])
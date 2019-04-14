import torch 


vector = torch.randn(1, 16)
print(vector)
vector = torch.norm(vector, p=2)
print(vector)
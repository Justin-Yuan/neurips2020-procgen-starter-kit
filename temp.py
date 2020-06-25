import torch 


a = torch.ones(2,2)
a.requires_grad=True

b = torch.ones(2,2)
b.requires_grad=True

c = a * b
d = torch.sum(c)
e = d.clone()

e.backward()
d.backward()


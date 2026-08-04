import torch
sd = torch.load('transduction_model.pt', map_location='cpu')
for k in sorted(sd.keys()):
    print(k, list(sd[k].shape), sd[k].dtype)

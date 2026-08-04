import torch
import sys

files = [
    'models_mamba1/transduction_model/model.pt',
    'V2/transduction_model.pt', 
    'V3/transduction_model.pt',
]
for pt in files:
    print('=====', pt, '=====')
    try:
        sd = torch.load(pt, map_location='cpu')
        for k in sorted(sd.keys()):
            print(k, list(sd[k].shape), sd[k].dtype)
    except Exception as e:
        print('ERROR:', e)
    print()

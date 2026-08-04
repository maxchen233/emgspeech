import torch
files = [
    'models_mamba1/transduction_model/model.pt',
    'models/transduction_model/model.pt',
]
for pt in files:
    print('=====', pt, '=====')
    try:
        sd = torch.load(pt, map_location='cpu')
        has_mamba = any('mamba' in k for k in sd.keys())
        print('has_mamba_keys:', has_mamba)
        for k in sorted(sd.keys()):
            if 'mamba' in k or k.startswith('conv_blocks') or k.startswith('w_') or k.startswith('final'):
                print(k, list(sd[k].shape), sd[k].dtype)
    except Exception as e:
        print('ERROR:', e)
    print()

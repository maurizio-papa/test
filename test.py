import torch
import torchvision.models as models

# Load the checkpoint
checkpoint = torch.load('/tesi/avion_pretrain_lavila_vitb_best.pt', map_location=torch.device('cpu'))

# If the checkpoint contains the state_dict of the model
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # The checkpoint directly contains the state_dict

model = torch.nn.Module()
model.load_state_dict(state_dict) # Replace YourModel with your model class

# Load the model parameters from the checkpoint

print(model)
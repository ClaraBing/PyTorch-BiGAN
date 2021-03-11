import os
import torch
from model import Encoder

def get_encoder(latent_dim, fckpt='', ker_size=11):
  E = Encoder(z_dim=latent_dim, first_filter_size=ker_size)
  if fckpt and os.path.exists(fckpt):

    ckpt = torch.load(fckpt)
    loaded_sd = ckpt['E']
    try:
      E.load_state_dict(loaded_sd)
    except:
      curr_params = E.state_dict()
      curr_keys = list(curr_params.keys())

      updated_params = {}
      for k,v in loaded_sd.items():
        if 'bn7' in k:
          newk = k.replace('bn7', 'conv7')
        else:
          newk = k
        if newk in curr_keys and loaded_sd[k].shape==curr_params[newk].shape:
          updated_params[newk] = v
        else:
          print('Failed to load:', k)
      curr_params.update(updated_params)
      E.load_state_dict(curr_params)
  return E


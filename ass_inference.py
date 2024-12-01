#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import torch
import torch.nn as nn
from ass_config import get_config, latest_weights_file_path
from ass_train import get_model, get_ds, run_validation
from ass_translate import translate


# In[ ]:


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = latest_weights_file_path(config)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


# In[ ]:


run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)


# In[ ]:


t = translate("Why do I need to translate this?")


# In[ ]:


t = translate(34)


# This document will follow the train file but I will explore and explain the key features

# Import all these moduels, need to ensure that the "home made" moduels are in the correct
# location for this to work
import json
from tqdm import tqdm
import numpy as np
import os

import pandas as pd

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from lstm import LSTM
from lstm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort, Arithmetic
from lstm.args import get_parser


# Unsure but this will expand and parse arguments for use in this file, this didnt work
# in the .ipynb files only in .py
args = get_parser().parse_args()

# these are the same task params as used in the copy task in the original
task_params = {
    "task": "copy",
    "controller_size": 100,
    "memory_units": 128,
    "memory_unit_size": 20,
    "num_heads": 1,
    "seq_width": 8,
    "min_seq_len": 3,
    "max_seq_len": 20
}

# this will generate the dataset of copy values, would be ideal to set up my tasks in this
# way for easy of use
dataset = Arithmetic(task_params)


lstm = LSTM(
    input_size=34,
    output_size=32,
    controller_size= 100,
    batch_size=1
    )

criterion = nn.BCELoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)
losses = []
errors = []



max_iters = 1000000

for inter in tqdm(range(max_iters)):

  data = dataset[inter]
  input, target = data['input'], data['target']
  input = input[None,:]

  optimizer.zero_grad()
  lstm.reset()


  for i in range(input.size()[1]):

    in_data = torch.unsqueeze(input[0][i], 0)
    in_data = in_data[None,:]
    lstm(in_data)

  out = torch.zeros(target.size())

  in_data = torch.unsqueeze(torch.zeros(input.size()[2]), 0)
  in_data = in_data[None,:]

  for i in range(target.size()[0]):
    out[i] = lstm(in_data)

  loss = criterion(out, target)
  losses.append(loss.item())
  loss.backward()

  nn.utils.clip_grad_value_(lstm.parameters(), 10)
  optimizer.step()

  binary_output = out.clone()
  binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

  # sequence prediction error is calculted in bits per sequence
  error = torch.sum(torch.abs(binary_output - target))
  errors.append(error.item())

  if (inter % 10000) == 0:
    print('Iteration: {} tLoss: {} Error in bits per sequence: {}'.format(iter, np.mean(losses), np.mean(errors)))
    #log_value('train_loss', np.mean(losses), iter)
    #log_value('bit_error_per_sequence', np.mean(errors), iter)



saved_loses = {'losses':losses,'errors':errors}
saved_loses = pd.DataFrame(saved_loses)



saved_model = 'saved_modelLSTM_arithmetic.pt'
cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, 'Saved_Models')

PATH_sl = os.path.join(PATH, 'saved_losses_LSTM_arithmetic.csv')
PATH = os.path.join(PATH, saved_model)


torch.save(lstm.state_dict(), PATH)
saved_loses.to_csv(PATH_sl)


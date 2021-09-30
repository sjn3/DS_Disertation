# This document will follow the train file but I will explore and explain the key features

# Import all these moduels, need to ensure that the "home made" moduels are in the correct
# location for this to work
import json
from tqdm import tqdm
import numpy as np
import os

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from lantm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser


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
    "min_seq_len": 1,
    "max_seq_len": 20
}

# this will generate the dataset of copy values, would be ideal to set up my tasks in this
# way for easy of use
dataset = CopyDataset(task_params)

# Establishment of the model
ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

# this will determin the loss gradient ect, can also use Adam
# BCELoss:  https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
# RMSprop:  https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
criterion = nn.BCELoss()
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)



args.num_iters = 200

losses = []
errors = []

for iter in tqdm(range(args.num_iters)): #tqdm is just a progress bar
    optimizer.zero_grad() # this sets the gradients to zero
    ntm.reset() # clears the model memory

    data = dataset[iter]  # will extract data depending on which iteration (will be random)
    input, target = data['input'], data['target']
    out = torch.zeros(target.size()) # sets the model output as equal to target size

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)

    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()

    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))
    errors.append(error.item())

    # ---logging---
    if iter % 200 == 0:
        print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        #log_value('train_loss', np.mean(losses), iter)
        #log_value('bit_error_per_sequence', np.mean(errors), iter)
        losses = []
        errors = []


args.saved_model = 'saved_model_copy.pt'
cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, 'Saved_Models')
PATH = os.path.join(PATH, args.saved_model)

torch.save(ntm.state_dict(), PATH)






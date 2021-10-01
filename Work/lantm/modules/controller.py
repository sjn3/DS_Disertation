import torch
from torch import nn
import torch.nn.functional as F

# The is no differece between the NTM controller and the LANTM controller

class LANTMController(nn.Module):

    def __init__(self, input_size, controller_size, output_size, read_data_size):
        super().__init__()
        self.input_size = input_size
        self.controller_size = controller_size
        self.output_size = output_size
        self.read_data_size = read_data_size

        self.controller_net = nn.LSTMCell(input_size, controller_size)
        self.out_net = nn.Linear(read_data_size, output_size)
        # nn.init.xavier_uniform_(self.out_net.weight)
        nn.init.kaiming_uniform_(self.out_net.weight)
        self.h_state = torch.zeros([1, controller_size])
        self.c_state = torch.zeros([1, controller_size])
        # layers to learn bias values for controller state reset
        self.h_bias_fc = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.h_bias_fc.weight)
        self.c_bias_fc = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.c_bias_fc.weight)
        self.reset()

    def forward(self, in_data, prev_reads):
        x = torch.cat([in_data] + prev_reads, dim=-1)
        self.h_state, self.c_state = self.controller_net(
            x, (self.h_state, self.c_state))
        
        
        
        return self.h_state, self.c_state

    def output(self, read_data):
        
        complete_state = torch.cat([self.h_state] + read_data, dim=-1)
        #complete_state = read_data.view(120,8)
        
        # used the Value Error function to test the different input and output dimensiosn
        #raise ValueError(len((self.out_net.weight[1])))
        
        # This is the section that caused the issue, I cant find the reason that the out_net refuses to process the data. All dimensions are the same as those in the NTM model
        # this issue shouldnt be occuring
        holder = self.out_net(complete_state)    #self.out_net(complete_state)
       
        
        output = F.sigmoid(holder)
        
        
        
        return output

    def reset(self, batch_size=1):
        in_data = torch.tensor([[0.]])  # dummy input
        h_bias = self.h_bias_fc(in_data)
        self.h_state = h_bias.repeat(batch_size, 1)
        c_bias = self.c_bias_fc(in_data)
        self.c_state = c_bias.repeat(batch_size, 1)

import torch
from torch import nn

from .modules.controller import LANTMController
from .modules.head_action import LANTMHead_Action
from .modules.memory import LANTMMemory


class LANTM(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 controller_size,
                 memory_units,
                 memory_unit_size,
                 num_heads):
        super().__init__()
        self.controller_size = controller_size
        
        # Establish the controller perameters
        self.controller = LANTMController(
            (input_size + (num_heads * memory_unit_size)), controller_size, output_size,
            read_data_size=(controller_size + (num_heads * memory_unit_size)))
        # Establish the Memory perameters and create the local memory
        self.memory = LANTMMemory(memory_units, memory_unit_size)
        self.memory_unit_size = memory_unit_size
        self.memory_units = memory_units
        
        # Establish the Head perameters
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [
                LANTMHead_Action('r', controller_size, key_size=memory_unit_size),
                LANTMHead_Action('w', controller_size, key_size=memory_unit_size)
            ]
        
        # Establish the local memory for each of the keys for the model
        self.prev_head_weights = []
        self.prev_action_weights = []        
        self.prev_reads = []

        
        self.reset()
    
    # Will need to reset the perameters every time the model is called
    def reset(self, batch_size=1):
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            prev_weight = torch.zeros([batch_size, self.memory.n])
            self.prev_head_weights.append(prev_weight)
        
        self.prev_action_weights = []
        for i in range(len(self.heads)):
            prev_act = torch.zeros([batch_size, self.memory.n])
            self.prev_action_weights.append(prev_act)
        
        self.prev_reads = []
        for i in range(self.num_heads):
            prev_read = torch.zeros([batch_size, self.memory.m])
            # using random initialization for previous reads
            nn.init.kaiming_uniform_(prev_read)
            self.prev_reads.append(prev_read)
            


    
    def forward(self, in_data):
        #first runs the controler
        controller_h_state, controller_c_state = self.controller(
            in_data, self.prev_reads)
        # establishes some local memory of differet factors
        read_data = []
        head_weights = []
        head_actions = []
        
        # Will run the read head and then the write head, the memory is imbeded in the heads code
        for head, prev_head_weight, prev_head_action in zip(self.heads, self.prev_head_weights, self.prev_action_weights):
            #This is the
            if head.mode == 'r':
                head_weight, r, _, _ = head(
                    controller_c_state, controller_h_state, prev_head_weight, prev_head_action, self.memory)
                read_data.append(r)
            else:
                head_weight, _, _, head_actions = head(
                    controller_c_state, controller_h_state, prev_head_weight, prev_head_action, self.memory)
            head_weights.append(head_weight)

        # Gives the model predictions
        output = self.controller.output(read_data)
        
        # Updates the local memory
        self.prev_head_weights = head_weights
        self.prev_action_weights = head_actions
        self.prev_reads = read_data

        return output

import torch
import torch.nn.functional as F
from torch import nn


class NTMHead_Action(nn.Module):

    def __init__(self, mode, controller_size, key_size):
        super().__init__()
        self.mode = mode
        self.key_size = key_size

        # all the fc layers to produce scalars for memory addressing
        self.key_fc = nn.Linear(controller_size, key_size)
        self.key_strength_fc = nn.Linear(controller_size, 1)

        # these five fc layers cannot be in controller class
        # since each head has its own parameters and scalars
        self.interpolation_gate_fc = nn.Linear(controller_size, 1)
        self.shift_weighting_fc = nn.Linear(controller_size, 3)
        self.sharpen_factor_fc = nn.Linear(controller_size, 1)
        # --(optional : for separation of add and erase mechanism)
        # self.erase_weight_fc = nn.Linear(controller_size, key_size)

        # fc layer to produce write data. data vector length=key_size
        self.write_data_fc = nn.Linear(controller_size, key_size)
        
        
        
        self.act_fc = nn.Linear(controller_size, key_size)
        self.act_strength_fc = nn.Linear(controller_size, 1)

        # these five fc layers cannot be in controller class
        # since each head has its own parameters and scalars
        self.interpolation_gateAct_fc = nn.Linear(controller_size, 1)
        self.shift_weightingAct_fc = nn.Linear(controller_size, 3)
        self.sharpen_factorAct_fc = nn.Linear(controller_size, 1)
        # --(optional : for separation of add and erase mechanism)
        # self.erase_weight_fc = nn.Linear(controller_size, key_size)
        
        
        
        self.reset()

    def forward(self, controller_state_c, controller_state_h, prev_weights, prev_action, memory, data=None):
        """Accept previous state (weights and memory) and controller state,
        produce attention weights for current read or write operation.
        Weights are produced by content-based and location-based addressing.

        Refer *Figure 2* in the paper to see how weights are produced.

        The head returns current weights useful for next time step, while
        it reads from or writes to ``memory`` based on its mode, using the
        ``data`` vector. ``data`` is filled and returned for read mode,
        returned as is for write mode.

        Refer *Section 3.1* for read mode and *Section 3.2* for write mode.

        Parameters
        ----------
        controller_state : torch.Tensor
            Long-term state of the controller.
            ``(batch_size, controller_size)``

        prev_weights : torch.Tensor
            Attention weights from previous time step.
            ``(batch_size, memory_units)``

        memory : ntm_modules.NTMMemory
            Memory Instance. Read write operations will be performed in place.

        data : torch.Tensor
            Depending upon the mode, this data vector will be used by memory.
            ``(batch_size, memory_unit_size)``

        Returns
        -------
        current_weights, data : torch.Tensor, torch.Tensor
            Current weights and data (filled in read operation else as it is).
            ``(batch_size, memory_units), (batch_size, memory_unit_size)``
        """

        # all these are marked as "controller outputs" in Figure 2
        key = self.key_fc(controller_state_c)
        b = F.softplus(self.key_strength_fc(controller_state_c))
        g = F.sigmoid(self.interpolation_gate_fc(controller_state_c))
        
        s = F.softmax(self.shift_weighting_fc(controller_state_c))
        # here the sharpening factor is less than 1 whereas as required in the
        # paper it should be greater than 1. hence adding 1.
        y = 1 + F.softplus(self.sharpen_factor_fc(controller_state_c))
        # e = F.sigmoid(self.erase_weight_fc(controller_state))  # erase vector
        a = self.write_data_fc(controller_state_c)  # add vector

        content_weights = memory.content_addressing(key, b)
        # location-based addressing - interpolate, shift, sharpen
        
        
        interpolated_weights = g * content_weights + (1 - g) * prev_weights
        shifted_weights = self._circular_conv1d(interpolated_weights, s)
        # the softmax introduces the exp of the argument which isn't there in
        # the paper. there it's just a simple normalization of the arguments.
        current_weights = shifted_weights ** y
        # current_weights = F.softmax(shifted_weights ** y)
        current_weights = torch.div(current_weights, torch.sum(
            current_weights, dim=1).view(-1, 1) + 1e-16)
        
        # This is the section where the key action will be defined
        # This vector undergoes a very similar to the key address but will 
        #   will impact the final memory store location vie Lie access interaction
        
        
        second_state = controller_state_c
        
        act = self.act_fc(second_state)
        b_act = F.softplus(self.act_strength_fc(second_state))
        g_act = F.sigmoid(self.interpolation_gateAct_fc(second_state))
        s_act = F.softmax(self.shift_weightingAct_fc(second_state))
        y_act = 1 + F.softplus(self.sharpen_factorAct_fc(second_state))

        
        
        content_action = memory.content_addressing(act, b_act)
        
        
        interpolated_action = g_act * content_action + (1 - g_act) * prev_action
        shifted_action = self._circular_conv1d(interpolated_action, s_act)
        current_action = shifted_action ** y_act
        current_action = torch.div(current_action, torch.sum(
            current_action, dim=1).view(-1, 1) + 1e-16)
        
        
        # This section will involve the interaction between the address and the
        #   action to give the final memory step/location for the weights to be
        #   stored.
        # This is done using Lie Access and then placed on a multi dimension
        #   memory space.
        
        
        output_weights = current_weights + current_action
        
        output_weights = torch.div(output_weights, torch.sum(
            output_weights, dim=1).view(-1, 1) + 1e-16)
        
        
        
        
        if self.mode == 'r':
            data = memory.read(output_weights)
        elif self.mode == 'w':
            # memory.write(current_weights, a, e)
            memory.write(output_weights, a)
        else:
            raise ValueError("mode must be read ('r') or write('w')")
        
            
        return output_weights, data, current_weights, current_action

    @staticmethod
    def _circular_conv1d(in_tensor, weights):
        # pad left with elements from right, and vice-versa
        batch_size = weights.size(0)
        pad = int((weights.size(1) - 1) / 2)
        in_tensor = torch.cat(
            [in_tensor[:, -pad:], in_tensor, in_tensor[:, :pad]], dim=1)
        out_tensor = F.conv1d(in_tensor.view(batch_size, 1, -1),
                              weights.view(batch_size, 1, -1))
        out_tensor = out_tensor.view(batch_size, -1)
        return out_tensor

    def reset(self):
        nn.init.xavier_uniform_(self.key_strength_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.interpolation_gate_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.shift_weighting_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.sharpen_factor_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.write_data_fc.weight, gain=1.4)
        # nn.init.xavier_uniform_(self.erase_weight_fc.weight, gain=1.4)

        # nn.init.kaiming_uniform_(self.key_strength_fc.weight)
        # nn.init.kaiming_uniform_(self.interpolation_gate_fc.weight)
        # nn.init.kaiming_uniform_(self.shift_weighting_fc.weight)
        # nn.init.kaiming_uniform_(self.sharpen_factor_fc.weight)
        # nn.init.kaiming_uniform_(self.write_data_fc.weight)
        # nn.init.kaiming_uniform_(self.erase_weight_fc.weight)

        nn.init.normal_(self.key_fc.bias, std=0.01)
        nn.init.normal_(self.key_strength_fc.bias, std=0.01)
        nn.init.normal_(self.interpolation_gate_fc.bias, std=0.01)
        nn.init.normal_(self.shift_weighting_fc.bias, std=0.01)
        nn.init.normal_(self.sharpen_factor_fc.bias, std=0.01)
        nn.init.normal_(self.write_data_fc.bias, std=0.01)
        # nn.init.normal_(self.erase_weight_fc.bias, std=0.01)



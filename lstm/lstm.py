import torch
from torch import nn



class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 controller_size,
                 batch_size):
        super().__init__()
        
        self.controller_size = controller_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.controller_net = nn.LSTMCell( self.input_size, self.controller_size)
        self.out_net = nn.Linear(self.controller_size, self.output_size)

        self.hx = torch.randn(self.batch_size, self.controller_size)
        self.cx = torch.randn(self.batch_size, self.controller_size)

    
    def reset(self):
      self.hx = torch.zeros(self.batch_size, self.controller_size)
      self.cx = torch.zeros(self.batch_size, self.controller_size)
    
    def forward(self, in_data):

      self.hx, self.cx = self.controller_net(in_data[0], (self.hx, self.cx))

      output = torch.sigmoid(self.out_net(self.hx))


      #output = []

      #for i in range(in_data.size()[0]):
        #self.hx, self.cx = self.controller_net(in_data[i], (self.hx, self.cx))
        #output.append(torch.sigmoid(self.out_net(hx)))

      return output
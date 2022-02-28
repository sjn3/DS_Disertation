import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial
import struct



class Arithmetic(Dataset):
    

    def __init__(self, task_params):
        """Initialize a dataset instance for arithmetic task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to copy task.
        """
        self.seq_width = 32
        self.min_seq_len = task_params['min_seq_len']
        self.max_seq_len = task_params['max_seq_len']

        self.arith_func = {'add':[1,0],'sub':[0,1],'mul':[1,1],'div':[0,0]}
        self.arith_keys = list(self.arith_func.keys())

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        
        # fist find the function that will take place
        arith_seq = torch.zeros(self.seq_width+2)
        arith_task = self.arith_keys[ int(torch.randint(0,4,(1,))) ]
        arith_key = self.arith_func[arith_task]
        arith_seq[:len(arith_key)] = torch.tensor(arith_key)
        arith_seq = arith_seq[None,:]

        # Next will determin the number of number that will be computed, this will need to be an even number which is >= 2
        seq_len = torch.randint(self.min_seq_len, self.max_seq_len, (1,))
        if seq_len%2 != 0:
          seq_len -= 1
        seq_len = int(float(seq_len) / 2)

        #this next section produces randomised the numbers (0-100) and then turns them into 32 binary code
        #the will be two 0 spaces infront of the sequence, this will allow the arith_seq to stand out
        first_num = torch.zeros(self.seq_width + 2)[None,:]
        for i in range(seq_len):
          num = float(torch.rand(1)) * 100
          num = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
          num = [0,0] + [int(i) for i in num]
          num = torch.tensor(num)[None,:]
          first_num = torch.cat((first_num, num))
        first_num = first_num[1:]
      

        second_num = torch.zeros(self.seq_width + 2)[None,:]
        for i in range(seq_len):
          num = float(torch.rand(1)) * 100
          num = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
          num = [0,0] + [int(i) for i in num]
          num = torch.tensor(num)[None,:]
          second_num = torch.cat((second_num, num))
        second_num = second_num[1:]
        
        # Merge the firs, arith and second to form an tensor of size(2*seq_len +1, 34)
        input = torch.cat((first_num, arith_seq, second_num))

        # The next part of the code is responisble for creating the target
        # First need to reverse the above process to get the float numbers again
        first_list = []
        for i in range( len(first_num) ):
          b_num = first_num[i].tolist()
          b_str = ''
          for s in b_num:
            b_str += str( int(s))
          b_int = int(b_str, 2)
          first_list.append(struct.unpack('f', struct.pack('I', b_int))[0])

        second_list = []
        for i in range( len(second_num) ):
          b_num = second_num[i].tolist()
          b_str = ''
          for s in b_num:
            b_str += str( int(s))
          b_int = int(b_str, 2)
          second_list.append(struct.unpack('f', struct.pack('I', b_int))[0])        
        
        # Will now perform the reliven arith function on the numbers ( [1,2] + [3,4] = [4,6])           
        if arith_task == 'add':
          target = [first_list[i] + second_list[i] for i in range( len(second_list) ) ]
        elif arith_task == 'sub':
          target = [first_list[i] - second_list[i] for i in range( len(second_list) ) ]
        elif arith_task == 'mul':
          target = [first_list[i] * second_list[i] for i in range( len(second_list) ) ]
        elif arith_task == 'div':
          target = [first_list[i] / second_list[i] for i in range( len(second_list) ) ]

        # Turn this target list into the 32 binary output,
        target_fin = torch.zeros(self.seq_width)[None,:]
        for i in target:
          num = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', i))
          num = [int(i) for i in num]
          num = torch.tensor(num)[None,:]
          target_fin = torch.cat((target_fin, num))
        target_fin = target_fin[1:]      

        

        # fill in input sequence, two bit longer and wider than target
        

        return {'input': input, 'target': target_fin}

from config import *

class PosnData(Dataset):
    def __init__(self, refs, num_samples=20000):
        self.refs = refs
        self.num_refs, self.input_size = refs.shape

        self.inputs = torch.rand(num_samples, self.input_size).to(DEVICE) * 2 - 1
        self.positions = (torch.matmul(self.inputs, self.refs.transpose(0,1)) > 0).float()
    
    def __getitem__(self, index):
        return self.inputs[index], self.positions[index]
    
    def __len__(self):
        return self.inputs.shape[0]
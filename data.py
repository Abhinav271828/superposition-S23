from config import *

def visualise(y, min, max, cmap='gray'):
    if (len(y.shape) == 1):
        y = y.unsqueeze(0)
    elif (len(y.shape) == 3):
        y = y.squeeze(0)
    #plt.imshow(y.detach().numpy(), cmap=cmap)
    #plt.colorbar()
    #plt.show()
    fig = px.imshow(y.detach().numpy(), zmin=min, zmax=max, color_continuous_scale=cmap)
    fig.show()

class OffsetData(Dataset):
    def __init__(self, input_size, offset, seq_length):
        self.input_size = input_size
        self.offset = offset

        self.inputs = torch.rand(20000, seq_length, input_size).to(DEVICE)
        self.outputs = torch.concat([torch.zeros(20000, offset, input_size).to(DEVICE),
                                    self.inputs[:, offset:, :]],
                                    dim=1).to(DEVICE)
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
    def __len__(self):
        return self.inputs.shape[0]

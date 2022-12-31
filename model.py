from data import *

class ToyModels(pl.LightningModule):
    def __init__(self, input_size, hidden_size, offset):
        super().__init__()
        self.save_hyperparameters()
        self.is_ = input_size
        self.hs = hidden_size
        self.offset = offset
        self.lr = 0.01
        self.lf = nn.MSELoss()
    
    def get_dataloader(self, seq_length=200, num_samples=20000):
        ds = OffsetData(self.is_, self.offset, seq_length, num_samples)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        return dl

    def step(self, loss_type, batch, batch_idx):
        hs, pred = self(batch[0])
        loss = self.lf(pred, batch[1])
        self.log(loss_type, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step("train_loss", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("val_loss", batch, batch_idx)

    def train_dataloader(self):
        return self.get_dataloader()

    def val_dataloader(self):
        return self.get_dataloader(num_samples=5000)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class ToyRNN(ToyModels):
    def __init__(self, input_size, hidden_size, offset, nonlinearity='relu', bias=True):
        super().__init__(input_size, hidden_size, offset)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True, nonlinearity=nonlinearity, bias=bias)
        self.ffn = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=input_size, bias=bias),
                                 nn.ReLU())
        self.model_type = "nonlinear"

    def forward(self, batch):
                      # [bz, seq, is]
        hidden_states, _ = self.rnn(batch)
        # [bz, seq, hs]
        preds = self.ffn(hidden_states)
        # [bz, seq, is]
        return hidden_states, preds

class LinRNN(ToyModels):
    def __init__(self, input_size, hidden_size, offset, bias=True):
        super().__init__(input_size, hidden_size, offset)
        self.w_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias)
        self.w_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)
        self.ffn = nn.Linear(in_features=hidden_size, out_features=input_size, bias=bias)
        self.model_type = "linear"
    
    def forward(self, batch):
                      # [bz, seq, is]
        bz, _, _ = batch.shape
        batch = batch.transpose(0,1)
        # [seq, bz, is]
        h = torch.zeros(bz, self.hs)
        hidden_states = []
        for timestep in batch:
            # [bz, is]
            h = self.w_x(timestep) + self.w_h(h)
            # [bz, hs]
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states, dim=1)
        # [bz, seq, hs]
        preds = self.ffn(hidden_states)
        # [bz, seq, is]
        return hidden_states, preds
    
class TieRNN(ToyModels):
    def __init__(self, input_size, hidden_size, offset, bias=False):
        super().__init__(input_size, hidden_size, offset)
        self.bias = bias
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True, nonlinearity='relu', bias=bool(bias))
        if bias == True: self.b_f = nn.Parameter(torch.rand(input_size))
        self.act = nn.ReLU()
        self.model_type = "tied"

    def forward(self, batch):
                      # [bz, seq, is]
        hidden_states, _ = self.rnn(batch)
        # [bz, seq, hs]
        if self.bias == False:
            preds = self.act(torch.matmul(hidden_states, self.rnn.weight_ih_l0))
            # [bz, seq, is]
        elif self.bias == True:
            preds = self.act(torch.matmul(hidden_states, self.rnn.weight_ih_l0) + self.b_f)
                           # [bz, seq, is]                                      # [is]
        elif self.bias == 'tied':
            preds = self.act(torch.matmul(hidden_states, self.rnn.weight_ih_l0)
                             # [bz, seq, is]
                             - torch.matmul(self.rnn.weight_ih_l0.transpose(0,1), (self.rnn.bias_ih_l0+self.rnn.bias_hh_l0)))
                                            # [is, hs]                            # [hs]
                             # [is]
        return hidden_states, preds
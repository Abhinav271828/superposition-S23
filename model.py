from data import *

class ToyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='relu'):
        super().__init__()
        self.is_ = input_size
        self.hs = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, nonlinearity=nonlinearity)
        self.ffn = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=input_size),
                                 nn.ReLU())

    def forward(self, batch):
                      # [bz, seq, is]
        hidden_states, _ = self.rnn(batch)
        # [bz, seq, hs]
        preds = self.ffn(hidden_states)
        # [bz, seq, is]
        return preds

    def train_epoch(self, dl, optim, lf):
        sum = 0
        for batch, gold in tqdm(dl):
            optim.zero_grad()
            pred = self(batch)
            loss = lf(pred, gold)
            loss.backward()
            optim.step()
            sum += loss.item()
        return sum/len(dl)

    def train(self, dataset, batch_size, lr):
        dl = DataLoader(dataset, batch_size=batch_size)
        optim = torch.optim.Adam(params=self.parameters(), lr=lr)
        lf = nn.MSELoss()
        for e in range(20):
            avg_loss = self.train_epoch(dl, optim, lf)
            print(f"Loss at epoch {e}: {avg_loss}")

class LinRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.is_ = input_size
        self.hs = hidden_size
        self.w_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.w_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.ffn = nn.Linear(in_features=hidden_size, out_features=input_size, bias=True)
    
    def forward(self, batch):
                      # [bz, seq, is]
        bz, _, _ = batch.shape
        batch = batch.transpose(0,1)
        # [seq, bz, is]
        h = torch.zeros(bz, self.hs)
        states = []
        for timestep in batch:
            # [bz, is]
            h = self.w_x(timestep) + self.w_h(h)
            # [bz, hs]
            states.append(h)
        states = torch.stack(states, dim=1)
        # [bz, seq, hs]
        preds = self.ffn(states)
        # [bz, seq, is]
        return preds
    
    def train_epoch(self, dl, optim, lf):
        sum = 0
        for batch, gold in tqdm(dl):
            optim.zero_grad()
            pred = self(batch)
            loss = lf(pred, gold)
            loss.backward()
            optim.step()
            sum += loss.item()
        return sum/len(dl)

    def train(self, dataset, batch_size, lr):
        dl = DataLoader(dataset, batch_size=batch_size)
        optim = torch.optim.Adam(params=self.parameters(), lr=lr)
        lf = nn.MSELoss()
        for e in range(20):
            avg_loss = self.train_epoch(dl, optim, lf)
            print(f"Loss at epoch {e}: {avg_loss}")
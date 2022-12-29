from data import *

class ToyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='relu', bias=True):
        super().__init__()
        self.is_ = input_size
        self.hs = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True, nonlinearity=nonlinearity, bias=bias)
        self.ffn = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=input_size, bias=bias),
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

    def train(self, dataset, lr=0.1):
        dl = DataLoader(dataset, batch_size=BATCH_SIZE)
        optim = torch.optim.SGD(params=self.parameters(), lr=lr)
        lf = nn.MSELoss()
        for e in range(10):
            avg_loss = self.train_epoch(dl, optim, lf)
            print(f"Loss at epoch {e}: {avg_loss}")

class LinRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.is_ = input_size
        self.hs = hidden_size
        self.w_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias)
        self.w_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)
        self.ffn = nn.Linear(in_features=hidden_size, out_features=input_size, bias=bias)
    
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

    def train(self, dataset, lr=0.1):
        dl = DataLoader(dataset, batch_size=BATCH_SIZE)
        optim = torch.optim.SGD(params=self.parameters(), lr=lr)
        lf = nn.MSELoss()
        for e in range(20):
            avg_loss = self.train_epoch(dl, optim, lf)
            print(f"Loss at epoch {e}: {avg_loss}")

class TieRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=None):
        super().__init__()
        self.is_ = input_size
        self.hs = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True, nonlinearity='relu', bias=False)

    def forward(self, batch):
                      # [bz, seq, is]
        hidden_states, _ = self.rnn(batch)
        # [bz, seq, hs]
        preds = torch.matmul(hidden_states, self.rnn.weight_ih_l0)
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

    def train(self, dataset, lr=0.1):
        dl = DataLoader(dataset, batch_size=BATCH_SIZE)
        optim = torch.optim.SGD(params=self.parameters(), lr=lr)
        lf = nn.MSELoss()
        for e in range(20):
            avg_loss = self.train_epoch(dl, optim, lf)
            print(f"Loss at epoch {e}: {avg_loss}")

def get_hidden_states(model, x):
    if isinstance(model, ToyRNN) or isinstance(model, TieRNN):
        hs, _ = model.rnn(x)
    elif isinstance(model, LinRNN):
        hs = []
        h = torch.zeros(model.hs)
        for t in x:
            h = model.w_x(t) + model.w_h(h)
            hs.append(h)
        hs = torch.stack(hs,dim=0)
    return hs
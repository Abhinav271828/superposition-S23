from data import *

class PosnModel(pl.LightningModule):
    def __init__(self, input_size, num_refs):
        super().__init__()
        self.save_hyperparameters(ignore=['lf'])
        self.input_size = input_size
        self.num_refs = num_refs

        self.refs = torch.rand(num_refs, input_size) * 2 - 1

        self.lf = nn.CrossEntropyLoss()
        self.lr = 0.1

        layers = []
        hidden = range(self.input_size, self.num_refs, -1)
        for i in range(len(hidden)-1):
            layers.append(nn.Linear(hidden[i], hidden[i+1], bias=True))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden[-1], self.num_refs, bias=True))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)
    
    def get_dataloader(self, num_samples=20000):
        ds = PosnData(self.refs, num_samples=num_samples)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        return dl

    def step(self, loss_type, batch, batch_idx):
        pred = self(batch[0])
        loss = self.lf(pred, batch[1])
        self.log(loss_type, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step("train_loss", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("val_loss", batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.step("test_loss", batch, batch_idx)

    def train_dataloader(self):
        return self.get_dataloader()

    def val_dataloader(self):
        return self.get_dataloader(num_samples=50000)
    
    def test_dataloader(self):
        return self.get_dataloader(num_samples=50000)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_acc(self):
        ds = PosnData(self.refs, num_samples=BATCH_SIZE)
        preds = (self(ds.inputs) > 0.5).float()
        return (preds == ds.positions).sum() / preds.numel()

    def forward(self, batch):
                      # [bz, n]
        return self.layers(batch)
        pass
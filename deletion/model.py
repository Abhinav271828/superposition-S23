from data import *
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pytorch_lightning import LightningModule

class GenderMLP(LightningModule):
    def __init__(self, dim=300):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 200), nn.Sigmoid(),
            nn.Linear(200, 100), nn.Sigmoid(),
            nn.Linear(100, 2)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-3

    def step(self, l_type, batch, batch_idx):
        words, genders = batch
        preds = self.mlp(words)

        loss = self.criterion(preds, genders)

        self.log(l_type, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step("train", batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step("val", batch, batch_idx)

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=self.lr)

    def train_dataloader(self):
        dataset = GenderBiasedWords(split = 'train')
        return DataLoader(dataset=dataset, batch_size=32)

    def val_dataloader(self):
        dataset = GenderBiasedWords(split = 'val')
        return DataLoader(dataset=dataset, batch_size=32)

    def test_dataloader(self):
        dataset = GenderBiasedWords(split = 'test')
        return DataLoader(dataset=dataset, batch_size=32)
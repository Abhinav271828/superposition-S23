from data import *
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
import os
from tqdm import tqdm

class GenderMLP(pl.LightningModule):
    def __init__(self, dim=300, layers=[200,100]):
        super().__init__()
        layers = [dim] + layers
        self.layers = [nn.Sequential(nn.Linear(layers[i], layers[i+1]), nn.Sigmoid())
                            for i in range(len(layers)-1)] + \
                      [nn.Linear(layers[-1], 3)]

        self.mlp = nn.Sequential(*self.layers)
        #self.proj = nn.Linear(layers[-1], 3)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-3
    
    def forward(self, words):
        logits = self.proj(self.mlp(words))
        return logits

    def step(self, l_type, batch, batch_idx):
        logits = self.forward(batch[0])

        loss = self.criterion(logits, batch[1])

        self.log(l_type, loss, prog_bar=True)
        return loss
    
    def get_acc(self, dl):
        correct = 0
        total = 0
        for batch in tqdm(dl):
            logits = self.forward(batch[0])
            preds = logits.argmax(dim=-1)

            correct += (preds == batch[1]).sum().item()
            total += preds.numel()
        
        print("Correct:", correct)
        print("Total:", total)
        return (correct/total)

    def training_step(self, batch, batch_idx):
        return self.step("train_loss", batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.step("val_loss", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step("val_loss", batch, batch_idx)

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=self.lr)

    def train_dataloader(self):
        dataset = GenderBiasedWords(split = 'train')
        return DataLoader(dataset=dataset, batch_size=32, num_workers=os.cpu_count())

    def val_dataloader(self):
        dataset = GenderBiasedWords(split = 'val')
        return DataLoader(dataset=dataset, batch_size=32, num_workers=os.cpu_count())

    def test_dataloader(self):
        dataset = GenderBiasedWords(split = 'test')
        return DataLoader(dataset=dataset, batch_size=32, num_workers=os.cpu_count())

def train_and_save(dim, layers):
    ckpt = pl.callbacks.ModelCheckpoint(dirpath='models/',
                                        filename=f"{'-'.join(str(l) for l in [dim]+layers)}",
                                        monitor='val_loss',
                                        mode='min',
                                        save_top_k=1)
    es = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    trn = pl.Trainer(auto_lr_find=True, max_epochs=5000, callbacks=[ckpt, es])
    model = GenderMLP(dim, layers)
    trn.tune(model)
    trn.fit(model)
    trn.test(model)
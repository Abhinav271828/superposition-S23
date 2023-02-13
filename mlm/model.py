from data import *

class RegModel(pl.LightningModule):
    def __init__(self, regex, num_layers, emb_dim, nhead, ff_dim, alphabet="0123abcd@M~"):
        super().__init__()
        self.save_hyperparameters(ignore=['lf'])
        self.lf = nn.CrossEntropyLoss()
        self.lr = 0.1

        self.regex = regex
        self.alphabet = alphabet

        self.embedding = nn.Embedding(len(alphabet), emb_dim)
        self.layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                                dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)
        self.decoder = nn.Linear(emb_dim, len(alphabet), bias=True)

    def get_dataloader(self, num_samples=20000):
        ds = RegData(self.regex, self.alphabet, num_samples=num_samples)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        return dl

    def step(self, loss_type, batch, batch_idx):
        strings, masked_strings, pad_masks = batch
        # [bz, sq]
        reps = self(masked_strings, pad_masks)
        # [bz, sq, d]
        preds = self.decoder(reps)
        # [bz, sq, v]

        masked_tokens = torch.masked_select(strings, masked_strings.eq(9)) # TODO: make 9 global MASK_IDX
        masked_preds = torch.masked_select(preds.flatten(0,1), (masked_strings.eq(9)
                                                                .unsqueeze(2).repeat(1,1,len(self.alphabet))
                                                                .view(-1, len(self.alphabet)))).view(-1,len(self.alphabet))

        loss = self.lf(masked_preds, masked_tokens)

        self.log(loss_type, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step("train_loss", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("val_loss", batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        strings, _, pad_masks = batch
        reps = self(strings, pad_masks)
        preds = self.decoder(reps)[:, :, :-1]

        loss = self.lf(preds.transpose(1,2), strings[:, :, 1:])
        
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def train_dataloader(self):
        return self.get_dataloader()

    def val_dataloader(self):
        return self.get_dataloader(num_samples=50000)
    
    def test_dataloader(self):
        return self.get_dataloader(num_samples=50000)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, strings, masks):
                      # [bz, n]
        embeddings = self.embedding(strings)
        representations = self.encoder(src=embeddings, src_key_padding_mask=masks)
        return representations
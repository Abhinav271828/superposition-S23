import torch
from torch import tensor
import torch.nn.functional as F
#from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torchtext.vocab import GloVe
from icecream import ic

class GenderBiasedWords(LightningDataModule):
    def __init__(self, split, name='6B', dim=300):
        self.embeddings = GloVe(name=name, dim=dim)

        gender = F.normalize((self.embeddings['he'] - self.embeddings['she']).unsqueeze(1))
        projections = F.normalize(self.embeddings.vectors, dim=1) @ gender
        # [V, 1]    = [V, d]                               @ [d, 1]
        projections = projections.squeeze(1)
        # [V]

        sorted_projections, sorted_indices = torch.sort(projections, descending=True)
        # [V]               [V]

        abs_projections = projections.abs()
        sorted_abs_projections, sorted_abs_indices = torch.sort(abs_projections)

        gendered_words = torch.concat([sorted_indices[:7500],
                                       sorted_indices[-7500:]], dim=0)
        # [15000]
        neutral_words = torch.masked_select(sorted_abs_indices,
                                            sorted_abs_projections.le(0.03))[:7500]

        words = torch.concat([gendered_words, neutral_words], dim=0)

        genders = tensor([0]*7500 + [1]*7500 + [2]*len(neutral_words))
        # [15000+n]

        associations = torch.stack([words, genders], dim=1)
        # [15000, 2]

        perm = torch.randperm(15000)
        shuffled_associations = torch.index_select(associations, dim=0, index=perm)
        # [15000, 2]

        breaks = int(0.49 * 15000), \
                 int(0.70 * 15000)
        if split == 'train':
            self.associations = shuffled_associations[:breaks[0]]
        elif split == 'val':
            self.associations = shuffled_associations[breaks[0]:breaks[1]]
        elif split == 'test':
            self.associations = shuffled_associations[breaks[1]:]
    
    def __getitem__(self, index):
        return self.embeddings.vectors[self.associations[index, 0]], \
               self.associations[index, 1]

    def __len__(self):
        return self.associations.shape[0]
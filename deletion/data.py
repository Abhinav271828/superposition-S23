import torch
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchtext.vocab import GloVe
from icecream import ic

class GenderBiasedWords(Dataset):
    def __init__(self, split, name='6B', dim=300):
        self.embeddings = GloVe(name=name, dim=dim)

        gender = F.normalize((self.embeddings['he'] - self.embeddings['she']).unsqueeze(1))
        projections = F.normalize(self.embeddings.vectors, dim=1) @ gender
        # [V, 1]    = [V, d]                               @ [d, 1]
        projections = projections.squeeze(1)
        # [V]

        sorted_projections, sorted_indices = torch.sort(projections, descending=True)
        # [V]               [V]

        gendered_words = torch.concat([sorted_indices[:7500], sorted_indices[-7500:]], dim=0)
        # [15000]
        genders = tensor([0]*7500 + [1]*7500)
        # [15000]

        associations = torch.stack([gendered_words, genders], dim=1)
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
        return self.associations[index, 0], self.associations[index, 1]

    def __len__(self):
        return self.associations.shape[0]
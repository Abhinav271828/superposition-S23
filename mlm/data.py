from config import *

class RegData(Dataset):
    def __init__(self, regex, alphabet="0123abcd@M~", num_samples=20000):
        self.regex = regex

        self.i2v = alphabet
        self.v2i = {c: i for i, c in enumerate(self.i2v)}

        strings = []
        masked_strings = []
        max_length = 0
        for _ in range(num_samples):
            sample = exrex.getone(self.regex)
            strings.append(sample)
            masked_strings.append(sample)

            max_length = max(max_length, len(sample))

            mask_posns = random.choices(range(len(sample)), k=ceil(len(sample)*0.15))
            for p in mask_posns:
                x = list(masked_strings[-1])
                x[p] = 'M'
                masked_strings[-1] = ''.join(x)

        self.strings = strings
        
        indices = []
        masked_indices = []
        pad_masks = []
        pad_index = self.v2i['~'] # = len(self.v2i) - 1
        for i in range(len(strings)):
            sample = strings[i]
            masked_sample = masked_strings[i]
            pad_length = max_length-len(sample)
            pad_masks.append([True]*len(sample) + [False]*pad_length)
            indices.append([self.v2i[c] for c in sample] + [pad_index]*pad_length)
            masked_indices.append([self.v2i[c] for c in masked_sample] + [pad_index]*pad_length)
        
        self.indices = tensor(indices)
        self.masked_indices = tensor(masked_indices)
        self.pad_masks = tensor(pad_masks).float()

    def __getitem__(self, index):
        return self.indices[index], self.masked_indices[index], self.pad_masks[index]
    
    def __len__(self):
        return self.indices.shape[0]
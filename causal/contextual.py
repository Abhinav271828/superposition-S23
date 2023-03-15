from transformers import BertModel, BertConfig, BertTokenizer
from datasets import load_dataset
import torch
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

LAYERS = range(1,13)
BATCH_SIZE = 32

bert = BertModel.from_pretrained('bert-base-uncased')
tokn = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


_, he_id, she_id, _ = tokn.encode(['he', 'she'])
he, she = (bert.embeddings(tensor([[he_id, she_id]]))
           # [1, 2, 768]
               .squeeze(0))
           # [2, 768]
gender = F.normalize(he - she, dim=0)
# [768]

class UnsupData(Dataset):
    def __init__(self):
        dataset = load_dataset("wikitext", "wikitext-103-v1")['test']

        sentences = []
        for item in dataset:
            ids = tokn.encode(item['text'])
            sentences.append(ids)
        
        max_len = 512
        for i in range(len(sentences)):
            l = len(sentences[i])
            sentences[i] = sentences[i] + ([0]*(max_len - l))  if l <= 512 else sentences[:512]
        
        self.sentences = sentences
    
    def __getitem__(self, index):
        return tensor(self.sentences[index])
    
    def __len__(self):
        return len(self.sentences)

dataset = UnsupData()
dl = DataLoader(dataset, batch_size=BATCH_SIZE)

#dataset = load_dataset("wikitext", "wikitext-103-v1")

for LAYER in LAYERS:
    f = open('proj{LAYER}.tsv', 'w')
    f.close()

emb_projections = []
hid_projections = [[] for _ in range(12)]
for input_ids in tqdm(dl):
    # [bz, msl]

    hidden_states = bert(input_ids=input_ids, output_hidden_states=True).hidden_states
    # 13 * [bz, msl, 768]

    embedding = hidden_states[0]
    e_proj = F.normalize(embedding, dim=1) @ gender
    e_proj = torch.masked_select(e_proj, input_ids.ne(0))
    emb_projections += e_proj.tolist()

    for LAYER in LAYERS:
        hiddens   = hidden_states[LAYER]
        # [bz, msl, 768]

        h_proj = F.normalize(hiddens,   dim=1) @ gender
        # [bz, msl]
        h_proj = torch.masked_select(h_proj, input_ids.ne(0))

        hid_projections[LAYER-1] += h_proj.tolist()
                               # [seq_len]

        f = open(f'proj{LAYER}.tsv', 'a')
        f.write(''.join(f"{e}\t{h}\n" for e, h in zip(emb_projections, hid_projections[LAYER-1])))
        f.close()

#for LAYER in LAYERS:
#    f = open(f'proj{LAYER}.tsv', 'w')
#    f.write('\n'.join(f"{e}\t{h}" for e, h in zip(emb_projections, hid_projections[LAYER-1])))
#    f.close()

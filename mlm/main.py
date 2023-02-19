from utils import *

regexes = ["(abcd)+",
           "(dab)+(320)+",
           "((acd)|(013))+",
           "(a|b|c|d)+(0|(2(0|1|3)+))"]

model_kwargs = {'num_layers': 4,
                'emb_dim': 10,
                'nhead': 2,
                'ff_dim': 10}

for i, regex in enumerate(regexes):
    mdl = RegModel(regex=regex, **model_kwargs, alphabet="0123abcd@M~")
    train_and_save(mdl, f"regex={regex}")
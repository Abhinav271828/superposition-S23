from utils import *

classes = [('nonlin', ToyRNN),
           ('tanh', ToyRNN),
           ('linear', LinRNN),
          ]

descriptions = []
for is_ in [7]:
    for hs in range(10,3,-1):
        for type in ['nonlin', 'tanh', 'linear']:
            for bias in [True, False]:
                for v in ['v1', 'v2']:
                    descriptions.append(f'in={is_}-out={hs}-{type}-bias={bias}-offset=0-{v}.ckpt')

def load_model_from_index(index):
    path = descriptions[index]
    model_type = dict(classes)[path.split('-')[2]]
    mdl = load_model_from_name(model_type, path)
    return mdl
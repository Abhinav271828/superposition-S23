from utils import *

classes = [('nonlin', ToyRNN),
           ('tanh', ToyRNN),
           ('linear', LinRNN),
          ]

descriptions = []
for is_ in [7, 8]:
    for hs in range(10,3,-1):
        for type in ['nonlin', 'tanh', 'linear']:
            for bias in [True, False]:
                for v in [1,2]:
                    descriptions.append(f'in={is_}-out={hs}-{type}-bias={bias}-offset=0-v{v}')

def load_model_from_index(index):
    path = descriptions[index] + '.ckpt'
    model_type = dict(classes)[path.split('-')[2]]
    mdl = load_model_from_name(model_type, path)
    return mdl

def create_model_from_index(index):
    path = descriptions[index]
    is_ = int(path.split('-')[0].split('=')[1])
    hs = int(path.split('-')[1].split('=')[1])
    bias = eval(path.split('-')[3].split('=')[1])
    offset = int(path.split('-')[4].split('=')[1])
    model_type = path.split('-')[2]
    cls = dict(classes)[model_type]
    if model_type == 'tanh':
        mdl = cls(input_size=is_, hidden_size=hs, bias=bias, offset=offset, nonlinearity='tanh')
    else:
        mdl = cls(input_size=is_, hidden_size=hs, bias=bias, offset=offset)
    
    train_and_save(mdl, descriptions[index])
    check_dimensionality(mdl)
    sv_box_plot(mdl)
    return mdl
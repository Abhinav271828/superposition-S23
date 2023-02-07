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
    #check_dimensionality(mdl)
    #sv_box_plot(mdl)
    return mdl

E = 269
lbs = []
ubs = []
x = OffsetData(7, 0, 1000, 1)[0][0]
for e in range(165, E):
    path = f"train_time_test-v{e}.ckpt"
    mdl = load_model_from_name(ToyRNN, path)

    w_x = mdl.rnn.weight_ih_l0.detach()
    hs, y = mdl(x)
    #hs = hs.view(-1, mdl.hs)
    #y = y.view(-1, mdl.is_)

    err = y - x
    cov_e = torch.cov(err.transpose(0,1))
    _, s_e, _ = svd(cov_e.detach())
    dim_e = len([s for s in s_e if s > 0.05])

    dim_h = mdl.is_ - dim_e

    _, s_M, _ = svd(w_x.transpose(0,1).detach())

    lb = (s_M[dim_h] if dim_h < mdl.hs else 0)/s_M[0]
    ub = (s_M[dim_h-1])/s_M[0]
    lbs.append(lb)
    ubs.append(ub)

line = (go.Figure().add_trace(go.Scatter(x=list(range(165,E)), y=lbs))
                   .add_trace(go.Scatter(x=list(range(165,E)), y=ubs)))
line.show()

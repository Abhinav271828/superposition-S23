from model import *

ds = OffsetData(7, 0, 200)

#toy = torch.load('models/in=7-out=5-linear-offset=0.pkl', map_location=DEVICE)

def visualise_model(model, dataset):
    x = dataset[0][0]
    if isinstance(model, ToyRNN):
        w_x = model.rnn.weight_ih_l0
        w_h = model.rnn.weight_hh_l0
        w_f = model.ffn[0].weight
        try: 
            b_x = model.rnn.bias_ih_l0
            b_h = model.rnn.bias_hh_l0
            b_f = model.ffn[0].bias
        except AttributeError:
            b_x = b_h = b_f = tensor(0)
        y = model(x)
    elif isinstance(model, LinRNN):
        w_x = model.w_x.weight
        w_h = model.w_h.weight
        w_f = model.ffn.weight
        b_x = model.w_x.bias if model.w_x.bias is not None else tensor(0)
        b_h = model.w_h.bias if model.w_h.bias is not None else tensor(0)
        b_f = model.ffn.bias if model.ffn.bias is not None else tensor(0)
        y = model(x.unsqueeze(0)).view(-1,model.is_)
    elif isinstance(model, TieRNN):
        w_x = model.rnn.weight_ih_l0
        w_h = model.rnn.weight_hh_l0
        w_f = w_x.transpose(0,1)
        try: 
            b_x = model.rnn.bias_ih_l0
            b_h = model.rnn.bias_hh_l0
            b_f = tensor(0)
        except AttributeError:
            b_x = b_h = b_f = tensor(0)
        y = model(x)

    hs = get_hidden_states(model, x)

    range_min = min(map(lambda t: t.min(), [w_x, b_x, w_h, b_h, w_f, b_f, x, hs, y])).item()
    range_max = max(map(lambda t: t.max(), [w_x, b_x, w_h, b_h, w_f, b_f, x, hs, y])).item()

    # Inputs-outputs
    visualise(x[:20].transpose(0,1), min=range_min, max=range_max, title="Inputs")
    visualise(hs[:20].transpose(0,1), min=range_min, max=range_max, title="Hidden states")
    visualise(y[:20].transpose(0,1), min=range_min, max=range_max, title="Outputs")

    # Basis vectors
    visualise(torch.mm(w_f,w_x), min=range_min, max=range_max, title="W_f • W_x")
    visualise(torch.mm(w_x.transpose(0,1),w_x), min=range_min, max=range_max, title="W_x^T • W_x")
    visualise(torch.norm(w_f, dim=0).unsqueeze(1), min=range_min, max=range_max, title="||W_f_i||", text_auto=True)
    visualise(torch.norm(w_x, dim=0).unsqueeze(1), min=range_min, max=range_max, title="||W_x_i||", text_auto=True)

    # Computation in latent space
    visualise(hs[0].unsqueeze(1), min=range_min, max=range_max, title="h_0")
    visualise(w_h, min=range_min, max=range_max, title="W_h")
    visualise(torch.mm(hs[0].unsqueeze(0), w_h.transpose(0,1)).transpose(0,1), min=range_min, max=range_max, title="W_h • h_0")
    if (len(b_h.shape) != 0):
        visualise((b_h+b_x).unsqueeze(1), min=range_min, max=range_max, title="b_h + b_x")
        visualise((torch.mm(hs[0].unsqueeze(0), w_h.transpose(0,1))+b_h+b_x).transpose(0,1), min=range_min, max=range_max, title="W_h • h_0 + b_h + b_x")

descriptions = [('7-10:Nonlin:True', 'models/in=7-out=10-nonlin-offset=0.pkl'),
                ('7-10:Nonlin:False', 'models/in=7-out=10-nonlin-offset=0-bias=False.pkl'),
                ('7-10:Linear:True', 'models/in=7-out=10-linear-offset=0.pkl'),
                ('7-10:Linear:False', 'models/in=7-out=10-linear-offset=0-bias=False.pkl'),
                ('7-10:Tied:True', 'models/in=7-out=10-tied-offset=0-bias=True.pkl'),
                ('7-10:Tied:False', 'models/in=7-out=10-tied-offset=0-bias=False.pkl'),
                ('7-10:Tied:Tied', 'models/in=7-out=10-tied-offset=0-bias=False.pkl'),

                ('7-7:Nonlin:True', 'models/in=7-out=7-nonlin-offset=0.pkl'),
                ('7-7:Nonlin:False', 'models/in=7-out=7-nonlin-offset=0-bias=False.pkl'),
                ('7-7:Linear:True', 'models/in=7-out=7-linear-offset=0.pkl'),
                ('7-7:Linear:False', 'models/in=7-out=7-linear-offset=0-bias=False.pkl'),
                ('7-7:Tied:True', 'models/in=7-out=7-tied-offset=0-bias=True.pkl'),
                ('7-7:Tied:False', 'models/in=7-out=7-tied-offset=0-bias=False.pkl'),
                ('7-7:Tied:Tied', 'models/in=7-out=7-tied-offset=0-bias=False.pkl'),

                ('7-5:Nonlin:True', 'models/in=7-out=5-nonlin-offset=0.pkl'),
                ('7-5:Nonlin:False', 'models/in=7-out=5-nonlin-offset=0-bias=False.pkl'),
                ('7-5:Linear:True', 'models/in=7-out=5-linear-offset=0.pkl'),
                ('7-5:Linear:False', 'models/in=7-out=5-linear-offset=0-bias=False.pkl'),
                ('7-5:Tied:True', 'models/in=7-out=5-tied-offset=0-bias=True.pkl'),
                ('7-5:Tied:False', 'models/in=7-out=5-tied-offset=0-bias=False.pkl'),
                ('7-5:Tied:Tied', 'models/in=7-out=5-tied-offset=0-bias=False.pkl')]

# Nonlinear RNN
## Excess space in hidden state
#toy = ToyRNN(7, 10).to(DEVICE)
#toy.train(ds)
#torch.save(toy, 'models/in=7-out=10-nonlin-offset=0.pkl')

## Just enough space in hidden state
#toy = ToyRNN(7, 7).to(DEVICE)
#toy.train(ds)
#torch.save(toy, 'models/in=7-out=7-nonlin-offset=0.pkl')

## Not enough space in hidden state
#toy = ToyRNN(7, 5).to(DEVICE)
#toy.train(ds)
#torch.save(toy, 'models/in=7-out=5-nonlin-offset=0.pkl')

# Fully linear RNN
## Excess space in hidden state
#toy = LinRNN(7, 10).to(DEVICE)
#toy.train(ds)
#torch.save(toy, 'models/in=7-out=10-linear-offset=0.pkl')

## Just enough space in hidden state
#toy = LinRNN(7, 7).to(DEVICE)
#toy.train(ds)
#torch.save(toy, 'models/in=7-out=7-linear-offset=0.pkl')

## Not enough space in hidden state
#toy = LinRNN(7, 5).to(DEVICE)
#toy.train(ds)
#torch.save(toy, 'models/in=7-out=5-linear-offset=0.pkl')
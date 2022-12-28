from model import *

ds = OffsetData(7, 0, 200)

#toy = torch.load('in=7-out=5-linear-offset=0.pkl', map_location=DEVICE)

def visualise_model(model, type, dataset):
    x = ds[0][0]
    if type == "nonlin":
        w_x = model.rnn.weight_ih_l0
        b_x = model.rnn.bias_ih_l0
        w_h = model.rnn.weight_hh_l0
        b_h = model.rnn.bias_hh_l0
        w_f = model.ffn[0].weight
        b_f = model.ffn[0].bias
        hs, _ = model.rnn(x)
        y = model(x)
    elif type == "linear":
        w_x = model.w_x.weight
        b_x = model.w_x.bias
        w_h = model.w_h.weight
        b_h = model.w_h.bias
        w_f = model.ffn.weight
        b_f = model.ffn.bias
        hs = []
        h = torch.zeros(model.hs)
        for t in x:
            h = model.w_x(t) + model.w_h(h)
            hs.append(h)
        hs = torch.stack(hs,dim=0)
        y = model(x.unsqueeze(0)).view(-1,model.is_)
    
    range_min = min(map(lambda t: t.min(), [w_x, b_x, w_h, b_h, w_f, b_f, x, hs, y])).item()
    range_max = max(map(lambda t: t.max(), [w_x, b_x, w_h, b_h, w_f, b_f, x, hs, y])).item()

    # Inputs-outputs
    visualise(x[:20].transpose(0,1), min=range_min, max=range_max)
    visualise(hs[:20].transpose(0,1), min=range_min, max=range_max)
    visualise(y[:20].transpose(0,1), min=range_min, max=range_max)

    # Basis vectors
    visualise(torch.mm(w_f,w_x), min=range_min, max=range_max)
    visualise(torch.mm(w_x.transpose(0,1),w_x), min=range_min, max=range_max)
    visualise(torch.norm(w_f, dim=0).unsqueeze(1), min=range_min, max=range_max)
    visualise(torch.norm(w_x, dim=0).unsqueeze(1), min=range_min, max=range_max)

    # Computation in latent space
    visualise(hs[0].unsqueeze(1), min=range_min, max=range_max)
    visualise(w_h, min=range_min, max=range_max)
    visualise(torch.mm(hs[0].unsqueeze(0), w_h.transpose(0,1)).transpose(0,1), min=range_min, max=range_max)
    visualise(b_x.unsqueeze(1), min=range_min, max=range_max)
    visualise((torch.mm(hs[0].unsqueeze(0), w_h.transpose(0,1))+b_x).transpose(0,1), min=range_min, max=range_max)
    visualise(b_h.unsqueeze(1), min=range_min, max=range_max)
    visualise((torch.mm(hs[0].unsqueeze(0), w_h.transpose(0,1))+b_h).transpose(0,1), min=range_min, max=range_max)
    visualise((b_h+b_x).unsqueeze(1), min=range_min, max=range_max)
    visualise((torch.mm(hs[0].unsqueeze(0), w_h.transpose(0,1))+b_h+b_x).transpose(0,1), min=range_min, max=range_max)

# Nonlinear RNN
## Excess space in hidden state
#toy = ToyRNN(7, 10).to(DEVICE)
#toy.train(ds, BATCH_SIZE, 0.01)
#torch.save(toy, 'in=7-out=10-nonlin-offset=0.pkl')

## Just enough space in hidden state
#toy = ToyRNN(7, 7).to(DEVICE)
#toy.train(ds, BATCH_SIZE, 0.01)
#torch.save(toy, 'in=7-out=7-nonlin-offset=0.pkl')

## Not enough space in hidden state
#toy = ToyRNN(7, 5).to(DEVICE)
#toy.train(ds, BATCH_SIZE, 0.01)
#torch.save(toy, 'in=7-out=5-nonlin-offset=0.pkl')

# Fully linear RNN
## Excess space in hidden state
#toy = LinRNN(7, 10).to(DEVICE)
#toy.train(ds, BATCH_SIZE, 0.01)
#torch.save(toy, 'in=7-out=10-linear-offset=0.pkl')

## Just enough space in hidden state
#toy = LinRNN(7, 7).to(DEVICE)
#toy.train(ds, BATCH_SIZE, 0.01)
#torch.save(toy, 'in=7-out=7-linear-offset=0.pkl')

## Not enough space in hidden state
#toy = LinRNN(7, 5).to(DEVICE)
#toy.train(ds, BATCH_SIZE, 0.01)
#torch.save(toy, 'in=7-out=5-linear-offset=0.pkl')
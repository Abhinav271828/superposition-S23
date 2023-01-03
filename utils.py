from model import *

def visualise(y, min, max, cmap=[(0,'blue'), (0.5, 'white'), (1, 'red')], title="", text_auto=False):
    if (len(y.shape) == 1):
        y = y.unsqueeze(0)
    elif (len(y.shape) == 3):
        y = y.squeeze(0)
    #plt.imshow(y.detach().numpy(), vmin=min, vmax=max, cmap=cmap)
    #plt.colorbar()
    #plt.show()
    fig = px.imshow(y.detach().numpy(), zmin=min, zmax=max, color_continuous_scale=cmap, title=title, text_auto=text_auto)
    fig.show()

def visualise_model(model):
    x = OffsetData(7, 0, 200, 1)[0][0]
    if model.model_type == "nonlin":
        w_x = model.rnn.weight_ih_l0
        w_h = model.rnn.weight_hh_l0
        w_f = model.ffn[0].weight
        try: 
            b_x = model.rnn.bias_ih_l0
            b_h = model.rnn.bias_hh_l0
            b_f = model.ffn[0].bias
        except AttributeError:
            b_x = b_h = b_f = tensor(0)
        hs, y = model(x)
    elif model.model_type == "linear":
        w_x = model.w_x.weight
        w_h = model.w_h.weight
        w_f = model.ffn.weight
        b_x = model.w_x.bias if model.w_x.bias is not None else tensor(0)
        b_h = model.w_h.bias if model.w_h.bias is not None else tensor(0)
        b_f = model.ffn.bias if model.ffn.bias is not None else tensor(0)
        hs, y = model(x.unsqueeze(0))
        hs = hs.view(-1, model.hs)
        y = y.view(-1, model.is_)
    elif model.model_type == "tied":
        w_x = model.rnn.weight_ih_l0
        w_h = model.rnn.weight_hh_l0
        w_f = w_x.transpose(0,1)
        try: 
            b_x = model.rnn.bias_ih_l0
            b_h = model.rnn.bias_hh_l0
            b_f = tensor(0)
        except AttributeError:
            b_x = b_h = b_f = tensor(0)
        hs, y = model(x)

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

def load_model_from_name(cls, name):
    checkpoint = torch.load('models/' + name)
    model = cls.load_from_checkpoint('models/' + name, checkpoint['hyper_parameters'])
    return model
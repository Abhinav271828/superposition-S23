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

def check_dimensionality(model, zero_out=None):
    x = OffsetData(7, 0, 1000, 1)[0][0]
    # [200, 7]

    if model.model_type == "nonlin":
        w_x = model.rnn.weight_ih_l0.detach()
        w_f = model.ffn[0].weight
        hs, y = model(x)
        if zero_out:
            for i in zero_out: w_x[i] = 0
    elif model.model_type == "linear":
        w_x = model.w_x.weight
        w_f = model.ffn.weight
        hs, y = model(x.unsqueeze(0))
        hs = hs.view(-1, model.hs)
        y = y.view(-1, model.is_)
    elif model.model_type == "tied":
        w_x = model.rnn.weight_ih_l0
        w_f = w_x.transpose(0,1)
        hs, y = model(x)

    range_min = min(map(lambda t: t.min(), [w_f, x, hs, y])).item()
    range_max = max(map(lambda t: t.max(), [w_f, x, hs, y])).item()

    visualise(x[:20].transpose(0,1), min=range_min, max=range_max, title="Inputs")
    visualise(hs[:20].transpose(0,1), min=range_min, max=range_max, title="Hidden states")
    visualise(y[:20].transpose(0,1), min=range_min, max=range_max, title="Outputs")
    
    cov_x = torch.cov(x.transpose(0,1))
    # [7, 7]          [7, 200]
    S1, _, _ = svd(cov_x.detach())
    # [7, 7] -> basis of input space
    S1 = tensor(S1)
    # can take S1 = torch.eye(7). it doesn't matter.

    S2 = w_x.transpose(0,1).detach()
    # [7, hs] -> basis of hidden space (ideal)
    _, s, _ = svd(S2)

    cov_h = torch.cov(hs.transpose(0,1))
    # [hs, hs]        [hs, 200]
    _, s_H, _ = svd(cov_h.detach())
    # [hs, hs] -> basis of hidden space (actual)
    visualise(tensor(s_H), 0, s_H.max(), title="Eigenvalues of H covariance")

    #s1ts1_i = tensor(inv(mm(S1.transpose(0,1), S1).detach()))
    ## [7, 7]
    #s1ts2   = mm(S1.transpose(0,1), S2)
    ## [7, hs]
    #s2ts2_i = tensor(inv(mm(S2.transpose(0,1), S2).detach()))
    ## [hs, hs]

    #M = mm(mm(mm(mm(S1, s1ts1_i), s1ts2), s2ts2_i), S2.transpose(0,1))
    ## [7, 7]        [7, 7] [7, 7] [7, hs] [hs, hs] [hs, 7]
    #_, s_M, _ = svd(M.detach())
    ## Eigenspace of M is intersection of input space and hidden space
    #visualise(tensor(s_M), 0, s_M.max(), title="Eigenvalues of Intersection Space")

    #M = torch.cov(S2.transpose(0,1))
    ## [hs, hs]    [hs, 7]
    ## Eigenvalues of M represent dims

    #_, s_M, _ = svd(M.detach())
    #visualise(tensor(s_M), 0, s_M.max(), title="Eigenvalues of W_x covariance")

    M = S2
    _, s_M, _ = svd(M.detach())
    visualise(tensor(s_M), 0, s_M.max(), title="Eigenvalues of W_x")

    err = y - x
    # [200, 7]
    cov_e = torch.cov(err.transpose(0,1))
    # [7, 7]          [7, 200]

    _, s_e, _ = svd(cov_e.detach())
    visualise(tensor(s_e), 0, s_M.max(), title="Eigenvalues of error space")

def load_model_from_name(cls, name):
    checkpoint = torch.load('models/' + name)
    model = cls.load_from_checkpoint('models/' + name, checkpoint['hyper_parameters'])
    return model
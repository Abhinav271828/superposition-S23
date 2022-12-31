from utils import *

classes = [('nonlin', ToyRNN),
           ('linear', LinRNN),
           ('tied', TieRNN)]

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

# General code to run, save, load and visualise model
#toy = ToyRNN(7, 10)
#ckpt = pl.callbacks.ModelCheckpoint(dirpath='models/', filename='in=7-out=10-nonlin-bias=True', monitor='val_loss', mode='min', save_top_k=1)
#es = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
#trn = pl.Trainer(auto_lr_find=True, max_epochs=100, callbacks=[ckpt, es])
#trn.tune(toy)
#trn.fit(toy)
#checkpoint = torch.load('models/in=7-out=10-nonlin-bias=True')
#toy_reloaded = ToyRNN.load_from_checkpoint('models/in=7-out=10-nonlin-bias=True',
#                                           checkpoint['hyper_parameters'])
#visualise_model(toy_reloaded, OffsetData(7, 0, 200, 1)[0][0])
from utils import *

classes = [('nonlin', ToyRNN),
           ('linear', LinRNN),
           ('tied', TieRNN)]

descriptions = [('7-10:Nonlin:True:v1', 'models/in=7-out=10-nonlin-bias=True-offset=0-v1.ckpt'),
                ('7-10:Nonlin:True:v2', 'models/in=7-out=10-nonlin-bias=True-offset=0-v2.ckpt'),
                ('7-10:Nonlin:False:v1', 'models/in=7-out=10-nonlin-bias=False-offset=0-v1.ckpt'),
                ('7-10:Nonlin:False:v2', 'models/in=7-out=10-nonlin-bias=False-offset=0-v2.ckpt'),
                ('7-10:Linear:True:v1', 'models/in=7-out=10-linear-bias=True-offset=0-v1.ckpt'),
                ('7-10:Linear:True:v2', 'models/in=7-out=10-linear-bias=True-offset=0-v2.ckpt'),
                ('7-10:Linear:False:v1', 'models/in=7-out=10-linear-bias=False-offset=0-v1.ckpt'),
                ('7-10:Linear:False:v2', 'models/in=7-out=10-linear-bias=False-offset=0-v2.ckpt'),
                ('7-10:Tied:True:v1', 'models/in=7-out=10-tied-bias=True-offset=0-v1.ckpt'),
                ('7-10:Tied:True:v2', 'models/in=7-out=10-tied-bias=True-offset=0-v2.ckpt'),
                ('7-10:Tied:False:v1', 'models/in=7-out=10-tied-bias=False-offset=0-v1.ckpt'),
                ('7-10:Tied:False:v2', 'models/in=7-out=10-tied-bias=False-offset=0-v2.ckpt'),

                ('7-7:Nonlin:True:v1', 'models/in=7-out=7-nonlin-bias=True-offset=0-v1.ckpt'),
                ('7-7:Nonlin:True:v2', 'models/in=7-out=7-nonlin-bias=True-offset=0-v2.ckpt'),
                ('7-7:Nonlin:False:v1', 'models/in=7-out=7-nonlin-bias=False-offset=0-v1.ckpt'),
                ('7-7:Nonlin:False:v2', 'models/in=7-out=7-nonlin-bias=False-offset=0-v2.ckpt'),
                ('7-7:Linear:True:v1', 'models/in=7-out=7-linear-bias=True-offset=0-v1.ckpt'),
                ('7-7:Linear:True:v2', 'models/in=7-out=7-linear-bias=True-offset=0-v2.ckpt'),
                ('7-7:Linear:False:v1', 'models/in=7-out=7-linear-bias=False-offset=0-v1.ckpt'),
                ('7-7:Linear:False:v2', 'models/in=7-out=7-linear-bias=False-offset=0-v2.ckpt'),
                ('7-7:Tied:True:v1', 'models/in=7-out=7-tied-bias=True-offset=0-v1.ckpt'),
                ('7-7:Tied:True:v2', 'models/in=7-out=7-tied-bias=True-offset=0-v2.ckpt'),
                ('7-7:Tied:False:v1', 'models/in=7-out=7-tied-bias=False-offset=0-v1.ckpt'),
                ('7-7:Tied:False:v2', 'models/in=7-out=7-tied-bias=False-offset=0-v2.ckpt'),

                ('7-5:Nonlin:True:v1', 'models/in=7-out=5-nonlin-bias=True-offset=0-v1.ckpt'),
                ('7-5:Nonlin:True:v2', 'models/in=7-out=5-nonlin-bias=True-offset=0-v2.ckpt'),
                ('7-5:Nonlin:False:v1', 'models/in=7-out=5-nonlin-bias=False-offset=0-v1.ckpt'),
                ('7-5:Nonlin:False:v2', 'models/in=7-out=5-nonlin-bias=False-offset=0-v2.ckpt'),
                ('7-5:Linear:True:v1', 'models/in=7-out=5-linear-bias=True-offset=0-v1.ckpt'),
                ('7-5:Linear:True:v2', 'models/in=7-out=5-linear-bias=True-offset=0-v2.ckpt'),
                ('7-5:Linear:False:v1', 'models/in=7-out=5-linear-bias=False-offset=0-v1.ckpt'),
                ('7-5:Linear:False:v2', 'models/in=7-out=5-linear-bias=False-offset=0-v2.ckpt'),
                ('7-5:Tied:True:v1', 'models/in=7-out=5-tied-bias=True-offset=0-v1.ckpt'),
                ('7-5:Tied:True:v2', 'models/in=7-out=5-tied-bias=True-offset=0-v2.ckpt'),
                ('7-5:Tied:False:v1', 'models/in=7-out=5-tied-bias=False-offset=0-v1.ckpt'),
                ('7-5:Tied:False:v2', 'models/in=7-out=5-tied-bias=False-offset=0-v2.ckpt')]

# General code to run, save, load and visualise model
#toy = ToyRNN(7, 10, 0)
#ckpt = pl.callbacks.ModelCheckpoint(dirpath='models/',
#                                    filename='in=7-out=10-nonlin-bias=True-offset=0',
#                                    monitor='val_loss',
#                                    mode='min',
#                                    save_top_k=1)
#es = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
#trn = pl.Trainer(auto_lr_find=True, max_epochs=100, callbacks=[ckpt, es])
#trn.tune(toy)
#trn.fit(toy)
#toy_reloaded = load_model_from_name(ToyRNN, 'in=7-out=10-nonlin-bias=True-offset=0')
#visualise_model(toy_reloaded)[0][0])
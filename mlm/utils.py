from model import *

def train_and_save(model, name):
    ckpt = pl.callbacks.ModelCheckpoint(dirpath='models/',
                                        filename=name,
                                        monitor='val_loss',
                                        mode='min',
                                        save_top_k=1)
    es = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    trn = pl.Trainer(auto_lr_find=True, max_epochs=5000, callbacks=[ckpt, es])
    trn.tune(model)
    trn.fit(model)
    trn.test(model)
    print("Accuracy is", model.get_acc())

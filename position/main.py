from utils import *

for i in range(13,21):
    for r in range(2, i):
        print(f"Training model of dimension {i} with {r} references")
        name = f"in={i}-refs={r}"
        train_and_save(PosnModel(i, r), name)

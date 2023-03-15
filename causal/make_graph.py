import matplotlib.pyplot as plt
from tqdm import tqdm

colours = [(x/12, x/12, 0.5) for x in range(0, 12)]
for layer in [1, 4, 8, 12]:
    f = open(f"proj{layer}.tsv", "r")
    e = []; h = []
    i = 0
    for line in tqdm(f):
      i += 1
      e.append(float(line.split('\t')[0]))
      h.append(float(line.split('\t')[1][:-1]))
      if i > 200000: break
    
    plt.scatter(x=e, y=h, color=colours[layer-1])
plt.show()

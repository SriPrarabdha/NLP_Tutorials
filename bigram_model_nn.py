import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt' , 'r').read().splitlines() 

chars = sorted(set(''.join(words)))
stoi = {s:i+1 for i , s in enumerate(chars)}
stoi['.'] = 0

xs , ys = [] , []

for w in words :
    ch = ['.'] + list(w) + ['.']

    for c1 , c2 in zip(ch , ch[1:]) :
        id1 = stoi[c1]
        id2 = stoi[c2]
        xs.append(id1)
        ys.append(id2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs , num_classes = 27)
yenc = F.one_hot(ys , num_classes = 27)

plt.imshow(xenc)
plt.show()
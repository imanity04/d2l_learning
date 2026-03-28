import random
import torch
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l

num_tosses = 1000
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])

fair_probs = torch.tensor([0.5,0.5])
x = Multinomial(100,fair_probs).sample()
print(x)
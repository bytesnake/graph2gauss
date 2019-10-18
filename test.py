import scipy.sparse as sp
from g2g.model import Graph2Gauss
from g2g.utils import load_dataset

g = load_dataset('data/cora_ml.npz')
A, X, z = g['A'], g['X'], g['z']

g2g = Graph2Gauss(A, X, L=6, verbose=True, max_iter=3)
sess = g2g.train()
g2g.test()

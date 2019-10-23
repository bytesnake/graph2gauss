import numpy as np
import scipy.sparse as sp
from g2g.model import Graph2Gauss
from g2g.utils import load_dataset

#g = load_dataset('data/cora_ml.npz')
#A, X, z = g['A'], g['X'], g['z']

file = np.load("/home/ec2-user/disambiguation-internship/data/graph.npz")
indices = file['links_indices']
indptr = file['links_indptr']
links = sp.csr_matrix((np.ones(len(indices)), indices, indptr))

attrs = file['attributes']

print(links.shape)
print(attrs.shape)

g2g = Graph2Gauss(A=links, X=attrs, L=16, verbose=True, max_iter=4, hidden_layers=[512])

sess = g2g.train()
#g2g.test()
g2g.save("/home/ec2-user/disambiguation-internship/data/saved_models/")

#out = g2g.predict(np.expand_dims(attrs, 1))
#np.save("out.npy", out)

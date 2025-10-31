import numpy as np
import os, pickle
from PIL import Image
from scipy.io import savemat
# from deepface import DeepFace

### READ FROM LAST SAVED PROGRESS AS A .pickle FILE
file_name = '../data/tsne-embeddings-lfw.pickle'

matrix = []
labels = []

with open(file_name, 'rb') as pfile:
    embeddings: dict = pickle.load(pfile)
    for item in embeddings:
        labels.append(item['person'])
        matrix.append(item['embedding'])

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html
savemat("labels.mat", {"labels": labels})
savemat("matrix.mat", {"matrix": np.array(matrix)})

print(len(labels))
print(len(set(labels))) # testing for uniqueness
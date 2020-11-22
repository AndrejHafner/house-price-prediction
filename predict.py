import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# # Load data
# df_train = pd.read_csv("data/train.csv")
# df_test = pd.read_csv("data/test.csv")
#
#
#
#
def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

mat = np.array([[0, 10,9,13],
                [10,0, 5,6],
                [9, 5, 0,0],
                [13,6, 0,0]])
searchable = mat.copy()
searchable[np.diag_indices_from(mat)] = np.iinfo(mat.dtype).max
x,y = np.where(searchable == np.min(searchable))

print(x,y)

# x, y = np.where(mat == np.min(mat[mask]))



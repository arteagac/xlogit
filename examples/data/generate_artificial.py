"""
This script creates artificial data for a discrete choice problem.
Assume there are three modes of transportation to choose from. Seven
variables were designed as significant and three as non-significant.
"""

import numpy as np
import pandas as pd


def noise(n_obs, perc=.2):
    noise_vec = np.zeros(n_obs)
    rand_pos = np.random.randint(n_obs, size=(int(n_obs*perc)))
    noise_vec[rand_pos] = np.random.normal(size=(int(n_obs*perc)))
    return noise_vec


np.random.seed(0)
N = 4000  # Number of observations
J = 3  # Number of alternatives

# Generate input data
df = pd.DataFrame()
df['id'] = np.repeat(np.arange(1, N+1), J)
df['alt'] = np.tile(np.array([1, 2, 3]), N)

df['price'] = np.tile([2, 3, 7], N) + noise(N*J)
df['time'] = np.tile([6, 3.5, 3], N) + noise(N*J)
df['conven'] = np.tile([3, 4, 8], N) + noise(N*J)
df['comfort'] = np.tile([4, 4.5, 2], N) + noise(N*J)
df['meals'] = np.tile([3, 2, 0], N) + noise(N*J)
df['petfr'] = np.tile([0, 0, 3], N) + noise(N*J)
df['emipp'] = np.tile([2, 0.5, 7], N) + noise(N*J)
df['nonsig1'] = np.tile([4, 2, 5], N) + noise(N*J)
df['nonsig2'] = np.tile([1, 3, 1], N) + noise(N*J)
df['nonsig3'] = np.tile([5, 2, 4], N) + noise(N*J)
df = df.round(3)

# Define coefficients (betas)
# Fixed betas
Bprice, Btime, Bconven, Bcomfort = -1, -1.5, 1, 1

# Random betas
Bmeals = np.random.normal(loc=2, scale=1, size=N)
Bpetfr = np.random.normal(loc=4, scale=1, size=N)
Bemipp = np.random.normal(loc=-2, scale=1, size=N)

# Convert betas to matrix for easy product
B = [np.repeat(Bprice, N*J), np.repeat(Btime, N*J), np.repeat(Bconven, N*J),
     np.repeat(Bcomfort, N*J), np.repeat(Bmeals, J), np.repeat(Bpetfr, J),
     np.repeat(Bemipp, J), np.zeros(N*J), np.zeros(N*J), np.zeros(N*J)]
B = np.vstack(B).T

# Multiply and generate probability
X = df.values[:, 2:]  # Extract only necessary columns
XB = (X*B).sum(axis=1).reshape(N, J)
eXB = np.exp(XB)
prob = eXB/eXB.sum(axis=1, keepdims=True)
# Use monte carlo simulation to predict choice
y = np.apply_along_axis(lambda p: np.eye(J)[np.random.choice(J, p=p)], 1, prob)
y = y.reshape(N*J,)
df['choice'] = y

# Save to CSV
df.to_csv("artificial.csv", index=False)

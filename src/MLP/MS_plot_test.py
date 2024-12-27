import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/test.csv', header=None)
ya = df.values

def ms_calculator(input_array):
    """
    Calculate ratios of running maximum/minimum to running sum for different moments of the input array.

    Args:
        input_array: Input array of numbers

    Returns:
        dict: Dictionary containing two arrays:
            'r': Ratios using running maximum
            'l': Ratios using running minimum
    """
    p = 4  # 1st, 2nd, 3rd and 4th moments
    input_array = input_array.flatten()
    # Initialize arrays to store results
    R_r = np.zeros((len(input_array), p))
    R_l = np.zeros((len(input_array), p))

    for i in range(p):
        # Raise each element to the power of the corresponding moment
        x = np.abs(input_array) ** (i + 1)

        # Calculate cumulative sum
        S = np.cumsum(x)

        # Calculate running maximum and minimum
        M_r = np.maximum.accumulate(x)
        M_l = np.minimum.accumulate(x)

        # Calculate ratios
        R_r[:, i] = M_r / S
        R_l[:, i] = M_l / S

    return {
        'r': R_r,
        'l': R_l
    }

results = ms_calculator(ya)


# make the plots
x = np.arange(1,10001)
#%%
fig = plt.figure()
plt.semilogx(x, results['l'][:,0])
plt.semilogx(x, results['l'][:,1])
plt.semilogx(x, results['l'][:,2])
plt.semilogx(x, results['l'][:,3])
plt.xlim(left=1)
plt.show()


fig = plt.figure()
plt.semilogx(x, results['r'][:,0])
plt.semilogx(x, results['r'][:,1])
plt.semilogx(x, results['r'][:,2])
plt.semilogx(x, results['r'][:,3])
plt.xlim(left=1)
plt.show()

# imports
from preprocessing import *

# Read in the data.
train, test = read_data();

# Smoothen out the lines.
smoothen(train)
smoothen(test)

# Bound the values from 1 to -1
normalize_bound(train)
normalize_bound(test)



# Simple plot of a random light curve after this processing
train.drop(["LABEL"], axis=1).iloc[1].plot()
plt.show()

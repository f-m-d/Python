import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labradors = 500

#Let's suppose that greyh are usually taller than labs
#But give it some randomness like in real life

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height= 24 + 4 * np.random.randn(labradors)

#Just showing the randomness of values obtained
plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()
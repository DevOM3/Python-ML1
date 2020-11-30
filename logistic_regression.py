import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

# print(iris.keys())  # get info about what's in iris

features = iris["data"][:, 3:]  # store the data at index 3
label = (iris["target"] == 2).astype(np.int)   # check if the label is equal to the second label and
# then convert it to 1 or 0 depending apon its bool for the classification


# training a logistic regression classifier
# creating reference
clf = LogisticRegression()
clf.fit(features, label)

# predicting the result
prediction = clf.predict(([[1.6]]))
print(prediction)


# using matplotlib to visualize the output
features_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # the linspace() will map values from 0 to 3 in 1000 points and
# then reshape will reshape it in n number of rows but a single column
label_probability = clf.predict_proba(features_new)    # predicting the probabilities from the x_new points

plt.plot(features_new, label_probability[:, 1], "g-", label="verginica")   # plotting the points on the graph
plt.show()

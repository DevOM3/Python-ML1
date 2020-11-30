from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier  # classifier name always starts with Capital letter as it is class

iris = datasets.load_iris()

# to get the description print(iris.DESCR)

features = iris.data    # .data contains features
label = iris.target     # .target is for labels


clf = KNeighborsClassifier()    # creating object of the KNeighbourClassifier
clf.fit(features, label)    # training our model

prediction = clf.predict([[1, 1, 1, 1]])    # testing the model
print(prediction)


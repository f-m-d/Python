from sklearn import tree

# Example of how machine learning works

# ML Uses "features" to descrive and reconize things.

#features = [[140,"smooth"], [130,"smooth"], [150,"bumpy"], [170,"bumpy"]]
features = [[140,1], [130,1], [150,0], [170,0]]

#labels=["apple", "apple", "orange", "orange"]
labels=[0,0,1,1]

#We need to train a classifier: A Decision Tree

clf=tree.DecisionTreeClassifier()

clf=clf.fit(features,labels)

print clf.predict([[150,0]])
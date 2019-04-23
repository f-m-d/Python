import numpy as np
import graphviz 
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Print the features of iris flowers
# and the iris flowers' names and
# Print labels and features

# print iris.feature_names
# print iris.target_names
# print iris.data[0]
# print iris.target[0]

# for i in range(len(iris.target)):
#    print "Example: %d , Label: %s, features %s" % (i, iris.target[i], iris.data[i])

#Training data
test_index=[0,50,100]
train_target=np.delete(iris.target, test_index)
train_data= np.delete(iris.data, test_index, axis=0)

#Testing data
test_target=iris.target[test_index]
test_data=iris.data[test_index]

#Training a classifier

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)



#Visualize tree


dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 


dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,   
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
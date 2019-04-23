from sklearn import tree

clf = tree.DecisionTreeClassifier()

#Height, Weight, Shoes Size

X = [ [181, 80, 44], [177,70,43], [160,60,38], [154,54,37],
[166,65,40], [190,90,47], [175,64,39], [177,70,40], [171,75,42],
[181,85,43]  ]

Y=['male','female','female','female','male','male','male','female','male', 'male']


clf.fit(X,Y)

prediction = clf.predict([[190, 70, 43]])

print ("The gender of 190/70/43 is: ")
print (prediction)
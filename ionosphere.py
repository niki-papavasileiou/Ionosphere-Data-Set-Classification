import pandas as pd
from google.colab import files
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


uploaded = files.upload()                                             #upload ionosphere_dataset.txt


for filename in uploaded.keys(): 
  #print(filename)
  lines = uploaded[filename].splitlines() # split the contents into lines
  ''' for line in lines:  # epanalipsi gia oles tin grammes
    print(line.decode('utf-8'))  # print the line meta apo kodikopoisi 
  '''

read = pd.read_csv(filename)


df = pd.DataFrame(read)   

labels_df = df.iloc[:, [34]]
features_df = df.iloc[:, 0:34] 


np_features = features_df.values
np_labels = labels_df.values.flatten() 
#print(np_labels)

length = len(np_features) 
#length

x1 = np_labels[np_labels=='g'].shape[0] 
x =(x1 * 100)/ length 
y1 = np_labels[np_labels=='b'].shape[0] 
y = (y1 *100)/ length

print("f good: %.1f%%,  f bad: %.1f%%" % (x,y))


train, test, train_labels, test_labels = train_test_split(np_features, np_labels, test_size=0.40,random_state=81)


dc_uniform = DummyClassifier(strategy="uniform")
dc_constant_g = DummyClassifier(strategy="constant", constant='g')
dc_constant_b = DummyClassifier(strategy="constant", constant='b')
dc_most_frequent = DummyClassifier(strategy="most_frequent")
dc_stratified = DummyClassifier(strategy="stratified")


model = dc_uniform.fit(train, train_labels)

pima_accuracy = {}
pima_accuracy['uniform (random)'] = dc_uniform.score(test, test_labels)
model = dc_constant_g.fit(train, train_labels)
pima_accuracy['constant g'] = dc_constant_g.score(test, test_labels)
model = dc_constant_b.fit(train, train_labels)
pima_accuracy['constant b'] = dc_constant_b.score(test, test_labels)
model = dc_most_frequent.fit(train, train_labels)
pima_accuracy['most frequent label'] = dc_most_frequent.score(test, test_labels)
model = dc_stratified.fit(train, train_labels)
pima_accuracy['stratified'] = dc_stratified.score(test, test_labels)


gnb = GaussianNB()


gnb.fit(train, train_labels)


gnb.score(test,test_labels)


pima_accuracy['gaussian naive bayes'] = gnb.score(test, test_labels)



neighbors = list(range(1,50,2))

cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train, train_labels, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())



mean_error = [1 - x for x in cv_scores]


plt.plot(neighbors, mean_error)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


optimal_k = neighbors[mean_error.index(min(mean_error))]
print("The optimal number of neighbors (calculated in the training set) is %d" % optimal_k)


knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(train, train_labels)
pred = knn.predict(test)
print("\nOptimal accuracy on the test set is", accuracy_score(test_labels, pred), "with k=", optimal_k)

dc_optimal_k = KNeighborsClassifier(n_neighbors = 5)
dc_optimal_k.fit(train, train_labels)

pima_accuracy['optimal k'] = dc_optimal_k.score(test, test_labels)

sorted_accuracy = [(k, pima_accuracy[k]) for k in sorted(pima_accuracy, key=pima_accuracy.get, reverse=False)]
for k, v in sorted_accuracy:
  print(k,v)

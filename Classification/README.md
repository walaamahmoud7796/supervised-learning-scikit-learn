# Using scikit-learn to fit a classifier 
``` python 
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(iris['data'],iris['target'])
X_new = np.array([[5.6,2.8,3.9,1.1],
                  [5.7,2.6,3.8,1.3],
                [4.7,3.2,1.3,0.2]])

prediction = knn.predict(X_new)
```
# Train/Test Split 
```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,# what propotion of the original data is used for the test set
                    random_state=21, # sets a seed for the random number generator that splits the data into train and test
                     stratify=y)# list or array containing the labels 

knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
y_pred

#check accuracy
knn.score(X_test,y_test)

```

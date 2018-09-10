import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('diabetes.data.txt', header = None, delimiter=' *, *', engine='python')
data.columns = ['num_pregent', 'glucose', 'bp', 'triceps', 'serum', 'bmi', 'dpf', 'age', 'class']
print(data.isnull().sum())
print(data.head(10))
data_rev = data
features = data.values[:,:3]
target = data.values[:,3]
print(target)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.33, random_state = 10)
clf = GaussianNB()
clf.fit(features, target)
target_pred = clf.predict(features_test)
target_pred = clf.predict([1,89,66,23,94,28.1,0.167,21])
print(target_pred)

target_pred = clf.predict([221, 73, 85])
print(target_pred)

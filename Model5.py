from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
file="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=read_csv(file,names=name)
array=dataframe.values
x=array[:,0:8]
y=array[:,8]
model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
#bagged descision tree like Random tree or Forest and Extra tree can be used to estimate the importance of features
#FEATURE TREE

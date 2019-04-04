from pandas import read_csv
from sklearn import tree
import numpy as np

file="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=read_csv(file,names=name)
print dataframe.feature_names
print dataframe.target_names
print dataframe.data[0]
print dataframe.target[0]
rm=[0,50,100]
new_target=np.delete(dataframe.target,rm)
new_data=np.delete(dataframe.data,rm,axis=0)
cf=tree.DecisionTreeClassifier()
cf=cf.fit(new_data,new_target)
p=cf.predict(dataframe.data[rm])
print "Original data",dataframe.target[rm]
print "Preddict Result",p

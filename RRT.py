from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
file="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(file,names=name)
array=data.values
x=array[:,0:8]
y=array[:,8]
n_split=5
seed=7
test_size=.33
kfold=ShuffleSplit(n_splits=n_split,test_size=test_size,random_state=seed)
model=LogisticRegression()
result=cross_val_score(model,x,y,cv=kfold)
print result.mean()*100,result.std()*100

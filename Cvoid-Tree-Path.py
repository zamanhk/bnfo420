from Bio import SeqIO

mySeq = []
for dna in SeqIO.parse("early jan parent strain.txt", "fasta"):
    mySeq.append(dna.seq)
    

X0 = []

#adding to X and y

for i in range(0,len(mySeq)-1):
    X0.append(mySeq[0].seq)


y0 = []
for j in range(1,len(mySeq)):
    y0.append(mySeq[i].seq)
    
from SequenceEncoding import encoding
# Encoding letters into numbers

X = []
for k in range(len(X0)):
    encoded_X = encoding(X0[k])
    X.append(encoded_X)
    
y = []
for l in range(len(y0)):
    encoded_y = encoding(y0[l])
    y.append(encoded_y)
    
# Fitting ML models and evaluating accuracy
from sklearn import tree
from sklearn import metrics

from sklearn import ensemble
rfr = ensemble.RandomForestRegressor(n_estimators=20)
rfr.fit(X,y)

rfrscores = cross_validation.cross_val_score(rfr,X,y,cv=2)
print('Random Forests',rfrscores)
print("Average Accuracy: %0.2f (+/- %0.2f)" % (rfrscores.mean()*100, rfrscores.std() *100))


X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.5,random_state=50)


rfr.fit(X_train,y_train)
print(rfr.score(X_test,y_test))

y_pred_rfr = rfr.predict(X_test)
print('Random Forests R2 score:', metrics.r2_score(y_test,y_pred_rfr,multioutput='variance_weighted'))
print('Random Forests MSE:', metrics.mean_squared_error(y_test,y_pred_rfr))

params = {'n_estimators': 199, 'max_depth': 20, 'min_samples_split': 10,
          'learning_rate': 0.01, 'loss': 'ls'}
          
gbr = ensemble.GradientBoostingRegressor(**params)

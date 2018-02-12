import numpy as np
import pandas as pd
import time
import sys,os
import psutil
from operator import itemgetter
	#------ Read the train and test data set from txt file
train = pd.read_csv('train.txt',header=None,delimiter=r"\s+")
traino=np.array(train)
train = traino[0::1]
if len(sys.argv) > 1:
	Testfile = sys.argv[1]
else:
	print("Please Provide the test.txt dataset as an argument")
	Testfile = 'validate.txt'

validate = pd.read_csv(Testfile,header=None,delimiter=r"\s+")
validate=np.array(validate[0::1])#traino[1::1]#
E = 0 
AccK = [0]
def dist(train,validate):
	process = psutil.Process(os.getpid())
	D2t = []
	W=0.1*np.identity(784)
	t0 = time.clock()
	for i in range(0,validate.shape[0]):
		'''
		D2 = []
		for j in range(0,train.shape[0]):
			D = train[j][0:-1] - validate[i][0:-1]
			D = np.sqrt(np.sum(np.square(D)))
			D2.append(D)
		'''
		V = validate[i][0:-1]*np.ones((train.shape[0],1))
		D = (train[:,0:-1]-V)#np.dot((train[:,0:-1]-V),W)#
		D2 = list(np.sqrt(np.sum(np.square(D),axis = 1)))
		D2z = zip(list(train[:,-1]),D2)#print len(D2)
		#print (D2z)
		D2z.sort(key=lambda x: x[1])
		D2t.append(D2z)
		#Knn = D2z[0:K]
	print(time.clock()-t0)
	print(process.memory_info().rss)
	return D2t

def KNN_Acc(D2z,validate,K):
	#print("Running ...")
	t0 = time.clock()
	E = 0
	#print len(D2z[0])
	for j in range(0,validate.shape[0]):
		Knn = D2z[j][0:K]
		Lp = [x[0] for x in Knn]
		Lr = validate[j][-1]
		Mv = np.bincount(Lp).argmax()-Lr
		if Mv != 0:
			E += 1
	Acc = float(np.linalg.norm(E))/validate.shape[0]#(1 - float(np.linalg.norm(E))/validate.shape[0])*100
	#print("Accuracy for " +str(K)+ "-nearest neighbours is equal:")
	#print("runtime = " + str(time.clock()-t0))
	return Acc
'''
K = 5
acc = KNN_Acc(train,validate,K)
print acc
'''
D2z = dist(train,validate)
for K in range(1,2*100,1): 
	acc = KNN_Acc(D2z,validate,K)
	AccK.append(acc)
#print(AccK)
import matplotlib.pyplot as plt
ww = train[1400,0:-1].reshape(28,28)
print(ww.shape)
#plt.imshow(ww.T)
plt.plot(AccK)
plt.show()


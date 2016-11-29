import numpy as np
import random
import gzip,cPickle
from sklearn.neighbors import KNeighborsClassifier
import time 

class KNearestNeighborsClassifier():

    def load_Data(self):
        f = gzip.open(r'E:\MNIST\data\mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()
        return (training_data, validation_data, test_data)

    def load_data_wrapper(self):

        tr_d, va_d, te_d = self.load_Data()
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_data = zip(training_inputs,tr_d[1])
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])
        return (training_data, validation_data, test_data)

    def vectorized_result(self, j):

        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

start_time=time.time()
obj=KNearestNeighborsClassifier()
training_data,validation_data, test_data = obj.load_data_wrapper()
neighbors=KNeighborsClassifier(n_neighbors=7)

#for i in xrange (30):
n=len(training_data)
revised_training_Data=[]
revised_training_result=[]
for x in training_data:
	revised_training_Data.append(x[0].flatten())
	revised_training_result.append(x[1])
	

neighbors.fit(revised_training_Data,revised_training_result)
final_results=[(neighbors.predict(np.reshape(x[0],(1,-1))),x[1]) for x in test_data]
print "Final Accuracy {0}%".format(sum( int (x==y) for x,y in final_results)/100.0)

print("--- %s seconds ---" % (time.time() - start_time))
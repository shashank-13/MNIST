import numpy as np
import random
import gzip,cPickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV

class SupportVectorMachine():

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


obj=SupportVectorMachine()
training_data,validation_data, test_data = obj.load_data_wrapper()

#clf=svm.SVC()

random.shuffle(validation_data)

temp_validation=[validation_data[k:k+10] for k in xrange(0,10000,10)]
temp_validation_data=[x[0][0].flatten() for x in temp_validation]
temp_validation_Result=[x[0][1] for x in temp_validation]


clk=svm.SVC()
parameters = {'kernel':['linear','rbf','poly'],'C':[1, 10,100,1000],'gamma':[1e-3, 1e-4,1e-5]}
clf = GridSearchCV(clk, parameters)
clf.fit(temp_validation_data,temp_validation_Result)
print (clf.best_params_)
clk=svm.SVC(**clf.best_params_)

n=len(training_data)
incrementer=10
for i in xrange (30):

    random.shuffle(training_data)
    temp_training=[training_data[k:k+incrementer] for k in xrange(0,n,incrementer)]
    revised_training_Data = [x[0][0].flatten() for x in temp_training]
    revised_training_result = [x[0][1] for x in temp_training]
    clk.fit(revised_training_Data,revised_training_result)
    final_results=[(clk.predict(np.reshape(x[0],(1,-1))),x[1]) for x in test_data]
    print "Epoch {0} : {1}%".format(i,sum( int (x==y) for x,y in final_results)/100.0)

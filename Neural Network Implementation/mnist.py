import random
import numpy as np
import gzip,cPickle
class Contain():

    def __init__(self,sizes):
        self.sizes=sizes
        self.total_layers=len(sizes)
        self.bias=[np.random.randn(x,1) for x in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]


    def feedForward(self,a):
        
        for x,y in zip(self.weights,self.bias):
            a=self.sigmoid(np.dot(x,a)+y)

        return a
            

    def stochasticGradient(self,training_data,epochs,eta,incrementer,test_data=None):
        if test_data:
            test_data_size=len(test_data)
        n=len(training_data)
        for i in xrange(epochs):
                random.shuffle(training_data)
                temp_training=[training_data[k:k+incrementer] for k in xrange(0,n,incrementer)]
                for p in temp_training:
                    self.update_function(p,eta)
                
                print " Epoch {0}: {1} / {2}".format(i,self.evaluate(test_data),test_data_size)



    def sigmoid(self,z):

        return 1.0/(1.0+np.exp(-z))

    def prime(self,x):

        return self.sigmoid(x) *(1 - self.sigmoid(x))

    def evaluate(self,test_data):

        test_results=[(np.argmax(self.feedForward(x)),y) for x,y in test_data ]
        
        return sum( int(x==y) for (x,y) in test_results)

    def update_function(self,train,eta):

        temp_weight=[np.zeros(w.shape) for w in self.weights]
        temp_bias=[np.zeros(b.shape) for b in self.bias]

        for p,q in train:

            delta_Bias,delta_Weight=self.backpropagate(p,q)
            temp_weight=[a+b for a,b in zip(temp_weight,delta_Weight)]
            temp_bias=[a+b for a,b in zip(temp_bias,delta_Bias)]


        self.weights=[w-(eta/len(train))*nw for w,nw in zip(self.weights,temp_weight)]
        self.bias=[w-(eta/len(train))*nw for w,nw in zip(self.bias,temp_bias)]




    def backpropagate(self,x,y):
        
        temp_weight = [np.zeros(w.shape) for w in self.weights]
        temp_bias = [np.zeros(b.shape) for b in self.bias]
        last_value=x
        complete_list=[]
        list_Activation=[x]
        
        for a,b in zip(self.weights,self.bias):
            z=np.dot(a,last_value)+b
            complete_list.append(z)
            last_value=self.sigmoid(z)
            list_Activation.append(last_value)

        
        delta=(list_Activation[-1]-y)*self.prime(complete_list[-1])
        temp_bias[-1]=delta
        temp_weight[-1]=np.dot(delta,list_Activation[-2].transpose())

        for l in xrange(2, self.total_layers):
            last_value= complete_list[-l]
            sp = self.prime(last_value)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            temp_bias[-l] = delta
            temp_weight[-l] = np.dot(delta, list_Activation[-l-1].transpose())
        return (temp_bias, temp_weight)

    def load_Data(self):
        f = gzip.open(r'E:\MNIST\data\mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()
        return (training_data, validation_data, test_data)


    def load_data_wrapper(self):

        tr_d, va_d, te_d = self.load_Data()
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [self.vectorized_result(y) for y in tr_d[1]]
        training_data = zip(training_inputs, training_results)
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])
        return (training_data, validation_data, test_data)

    def vectorized_result(self,j):
    
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

        
            

        
obj=Contain([784,30,10])
training_data,validation_data,test_data=obj.load_data_wrapper()
obj.stochasticGradient(training_data,30,3.0,10,test_data=test_data)


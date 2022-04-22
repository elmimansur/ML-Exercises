import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.cm as cm
import time


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Reshaping and rescaling x
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train2 = x_train.reshape(60000,784)
x_test2 = x_test.reshape(10000,784)


#One-hot-encoded versions of the labels
y_train2 = keras.utils.to_categorical(y_train, num_classes=10)
y_test2 = keras.utils.to_categorical(y_test, num_classes=10)

normalised = preprocessing.scale(x_train2)

pca = PCA()
pca.fit(normalised)
plt.plot(pca.explained_variance_ratio_)
plt.title("PVE vs Number of Principal Components")
plt.xlabel('number of principal component')
plt.ylabel('proportion of variance explained')
#plt.savefig('2a1.png',bbox_inches='tight')
plt.show()


X_transformed = pca.transform(normalised)
trans= pd.DataFrame(X_transformed)
trans["label"]=y_train
sns.FacetGrid(trans, height = 8).map(sns.scatterplot, 0, 1).add_legend()
plt.title("First Two Principal Components Without Label Colors")
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.savefig('2a2.png',bbox_inches='tight')
plt.show()


sns.FacetGrid(trans, hue = 'label', height = 8).map(sns.scatterplot, 0, 1).add_legend()
plt.title("First Two Principal Components With Label Colors")
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.savefig('2a3.png',bbox_inches='tight')
plt.show()

sns.FacetGrid(trans, hue = 'label', height = 8).map(sns.scatterplot, 0, 2).add_legend()
plt.title("First and Third Principal Components With Label Colors")
plt.xlabel('PC1')
plt.ylabel('PC3')
#plt.savefig('2a5.png',bbox_inches='tight')
plt.show()

sns.FacetGrid(trans, hue = 'label', height = 8).map(sns.scatterplot, 1, 2).add_legend()
plt.title("Second and Third Principal Components With Label Colors")
plt.xlabel('PC2')
plt.ylabel('PC3')
#plt.savefig('2a6.png',bbox_inches='tight')
plt.show()

# 3D
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
scatter= ax.scatter(trans.loc[:,0], trans.loc[:,1], trans.loc[:,2], c=trans.loc[:,"label"], s=6,cmap=cm.tab10,alpha=0.4)
ax.set_title("First Three Principal Components With Label Colors")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#plt.savefig('2a4.png',bbox_inches='tight')
plt.show()




# Random sampling from data (%10)
idx = np.random.choice(x_train2.shape[0], 6000, replace=False)
s_xtrain=x_train2[idx, :]
s_ytrain=y_train[idx]

idx2 = np.random.choice(x_test2.shape[0], 1000, replace=False)
s_xtest=x_test2[idx2, :]
s_ytest=y_test[idx2]

#Linear SVC
model = SVC(kernel='linear')
model.fit(s_xtrain, s_ytrain)
print('Accuracy on test set:',model.score(s_xtest,s_ytest))

#Poly kernel
model2 = SVC(kernel = 'poly', degree = 2)
model2.fit(s_xtrain, s_ytrain)
print('Accuracy on test set:',model2.score(s_xtest,s_ytest))

#RBF kernel

#1st RGB: Default hyperparameters
model3 = SVC(kernel = 'rbf',C=1,gamma=(1 / (s_xtrain.shape[1] * s_xtrain.var())))
model3.fit(s_xtrain, s_ytrain)
print('Accuracy on test set:',model3.score(s_xtest,s_ytest))

#2nd RGB: Increased C
model4 = SVC(kernel = 'rbf',C=15,gamma=(1 / (s_xtrain.shape[1] * s_xtrain.var())))
model4.fit(s_xtrain, s_ytrain)
print('Accuracy on test set:',model4.score(s_xtest,s_ytest))

#3rd RGB: Increased gamma
model5 = SVC(kernel = 'rbf',C=15,gamma=(1 / (s_xtrain.shape[1] * s_xtrain.var()))+0.001)
model5.fit(s_xtrain, s_ytrain)
print('Accuracy on test set:',model5.score(s_xtest,s_ytest))




#Neural Network
class NeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        self.params = self.initial()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initial(self):
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        output_layer=self.sizes[2]
        
        # initializing parameters 
        params = { 'W1':np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / hidden_layer),
            'W2':np.random.randn(output_layer, hidden_layer) * np.sqrt(1. / output_layer),
            'B1': np.zeros([hidden_layer,]),
            'B2' : np.zeros([output_layer,])}

        return params

    def forward_pass(self, x_tr):
        params = self.params

        params['A0'] = x_tr

        # input layer to hidden layer
        params['Z1'] = np.dot(params["W1"], params['A0'])+params['B1']
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer to output layer
        params['Z2'] = np.dot(params["W2"], params['A1'])+params['B2']
        params['A2'] = self.softmax(params['Z2'])

        return params['A2']

    def backward_pass(self, y_tr, output):
        params = self.params
        delta = {}

        # Calculate W2 update
        error = 2 * ((output - y_tr) / output.shape[0]) * self.softmax(params['Z2'], derivative=True)
        delta['W2'] = np.outer(error, params['A1'])

        # Calculate B2 update
        delta['B2']= error

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        delta['W1'] = np.outer(error, params['A0'])

        # Calculate B1 update
        delta['B1']= error

        return delta

    def update_network_parameters(self, delta):      
        for key, value in delta.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_test, y_test):        
        predictions = []

        for x, y in zip(x_test, y_test):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)
    
    def predict(self,x):
        self.forward_pass(x)
        return self.params['A2']

    def train(self, x_tr, y_tr, x_test, y_test):
        for iteration in range(self.epochs):
            start_time = time.time()
            for x,y in zip(x_tr, y_tr):
                output = self.forward_pass(x)
                delta = self.backward_pass(y, output)
                self.update_network_parameters(delta)
            
            accuracy = self.compute_accuracy(x_test, y_test)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))

nn = NeuralNetwork(sizes=[784, 100, 10])
nn.train(x_train2, y_train2, x_test2, y_test2)


predicted = np.zeros(y_test.shape)
for i in range(len(predicted)):
    predicted[i] = np.argmax(nn.predict(np.array(x_test2[i])))

predicted_class = np.round(predicted)

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(y_test, predicted_class)])

# Calculate the accuracy
accuracy = match_count / len(y_test)
# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))


#(Hansen & iuvenis, 2020)


# Neural network without regularization 
model = keras.Sequential()
model.add(layers.Dense(units=1200, activation='relu',input_dim=784))
model.add(layers.Dense(units=1200, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=["accuracy"])
Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
history = model.fit(x_train2, y_train2, epochs=10, batch_size=64, verbose=1, validation_split = 0.2, callbacks=[Early_Stop])

# Regularisation with dropout and Adam as optimizer
model2 = keras.Sequential()
model2.add(layers.Dense(units=1200, activation='relu',input_dim=784))
model2.add(layers.Dense(units=1200, activation='relu'))
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(units=10, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
Early_Stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
history = model2.fit(x_train2, y_train2, epochs=10, batch_size=256, verbose=1, validation_split = 0.2, callbacks=[Early_Stop])

yprobas = model2.predict(x_test2, batch_size=256)
ypred = yprobas.argmax(axis=-1)
test_acc = np.mean(np.equal(ypred, y_test))
print('Test set accuracy:', test_acc)

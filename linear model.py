import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import linear_model
import matplotlib.pyplot as plt


print("loading data...")
X_data=np.load('X_train_surge.npz')

slp = X_data['slp']                         #(5599, 40, 41, 41)
surge1_input = X_data['surge1_input']       #(5599, 10)
surge2_input = X_data['surge2_input']       #(5599, 10)

Y_all = pd.read_csv('Y_train_surge.csv').to_numpy()[:,1:]

print("normalizing data...")
pressure =slp.reshape(5599,40*41*41)
moy_p=101325
pressure-=moy_p
pressure /= 10000 # Normalize data to [-1, 1] range


print("shaping data...")
total_data=np.concatenate((pressure, surge1_input, surge2_input), axis=1) 

print("computing PCA...")
pca = PCA(n_components=170)   
pca.fit(total_data)       #PCA on train+test
print("applying PCA...")
in_data=pca.transform(total_data) 
in_dim=in_data.shape[1]


regr = linear_model.LinearRegression()   
regr.fit(in_data[:5000], Y_all[:5000])    #training on 5000 first


w = np.linspace(1, 0.1, 10)
w = np.concatenate((w,w))[np.newaxis]
def loss_np(y_true, y_pred):
    return (w*(y_true-y_pred)**2).mean(axis=-1)


print("loss : ",loss_np(Y_all[5000:],regr.predict(in_data[5000:])).mean())   #testing on 599 last


def plot_predict(i):
    y_pred=regr.predict(in_data[i].reshape(1,in_dim))[0]
    plt.figure()
    ax=plt.subplot(1,2,1)
    ax.plot(y_pred[:10], label="pred")
    ax.plot(Y_all[i][:10], label="true")
    plt.legend()
    ax=plt.subplot(1,2,2)
    ax.plot(y_pred[10:], label="pred")
    ax.plot(Y_all[i][10:], label="true")
    plt.legend()




import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import linear_model
import matplotlib.pyplot as plt


print("loading data...")
X_data_train=np.load('X_train_surge.npz')
slp_train = X_data_train['slp']                         #(5599, 40, 41, 41)
surge1_input_train = X_data_train['surge1_input']       #(5599, 10)
surge2_input_train = X_data_train['surge2_input']       #(5599, 10)

X_data_test=np.load('X_test_surge.npz')
slp_test = X_data_test['slp']                         #(5599, 40, 41, 41)
surge1_input_test = X_data_test['surge1_input']       #(5599, 10)
surge2_input_test = X_data_test['surge2_input']       #(5599, 10)


moy_p=101325
pressure_train = slp_train.reshape(5599,40*41*41)
pressure_train -= moy_p
pressure_train /= 10000 # Normalize data to [-1, 1] range

pressure_test = slp_test.reshape(509,40*41*41)
pressure_test -= moy_p
pressure_test /= 10000 # Normalize data to [-1, 1] range


X_train=np.concatenate((pressure_train, surge1_input_train, surge2_input_train), axis=1) 
X_test=np.concatenate((pressure_test, surge1_input_test, surge2_input_test), axis=1) 

print("computing PCA...")
pca = PCA(n_components = 170)   
pca.fit(np.concatenate((X_train, X_test)))       #PCA on train+test
print("applying PCA...")
X_train_pca = pca.transform(X_train) 
X_test_pca = pca.transform(X_test)

Y_train = pd.read_csv('Y_train_surge.csv').to_numpy()[:,1:]


regr = linear_model.LinearRegression()   
regr.fit(X_train_pca, Y_train)   


w = np.linspace(1, 0.1, 10)
w = np.concatenate((w,w))[np.newaxis]
def loss_np(y_true, y_pred):
    return 2*(w*(y_true-y_pred)**2).mean(axis=-1)

train_losses = loss_np(Y_train, regr.predict(X_train_pca))
print("train loss : ", train_losses.mean()) 

plt.hist(train_losses, bins=200)


y_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]
def write_submission():
    y_test_benchmark = pd.DataFrame(data=regr.predict(X_test_pca), columns=y_columns, index=X_data_test["id_sequence"])
    y_test_benchmark.to_csv('linear_model_submission.csv', index_label='id_sequence', sep=',')


def plot_predict(i):
    y_pred=regr.predict(X_train_pca[i].reshape(1, pca.n_components))[0]
    plt.figure()
    ax=plt.subplot(1,2,1)
    ax.plot(y_pred[:10], label="pred")
    ax.plot(Y_train[i][:10], label="true")
    plt.legend()
    ax=plt.subplot(1,2,2)
    ax.plot(y_pred[10:], label="pred")
    ax.plot(Y_train[i][10:], label="true")
    plt.legend()


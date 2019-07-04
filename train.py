
import numpy as np
import pandas as pd
import keras.backend as K
from keras import regularizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()

# onehotcode to address the features that do not belong to number 
def onehotcode(X):
	ele = np.unique(X)
	X_hot = np.zeros((len(X),len(ele)))
	for i in range(len(ele)):
		X_hot[X==ele[i],i] = 1
	return X_hot

def process(X):
  # process X into number feature using one hot coding 
	X[:,0] = labelencoder_x.fit_transform(X[:,0])
	X[:,-2] = labelencoder_x.fit_transform(X[:,-2])

	X_hot = onehotcode(X[:,0])
	X = X[:,1:]
	X = np.concatenate((X,X_hot),axis = 1)
	X = X[:,[0,1,2,3,5,6,7]]
	return X


dataframe = pd.read_csv('./train_cleaned.csv')
print(dataframe.head())
dataset = dataframe.values
dataset = dataset[:,[5,7,10,12,13,14,15,16,17]]
dataset = dataset[dataset[:,-1]>0,:]
X = dataset[:,:-2]
Y = dataset[:,-1:]

# the process step is necessary
X= process(X)
# normailzation
max_X = np.max(X,axis = 0)
X= X/max_X
Y= Y/20000.
input_dimen = len(max_X)

def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(40, input_dim=input_dimen, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))
	model.add(Dense(20, kernel_initializer='normal',activation='relu'))	
	model.add(Dense(16, kernel_initializer='normal',activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
model = larger_model()

model.summary()
from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
# train
model.fit(
  X,
  Y,
	batch_size= 128,
  epochs = 150,
  validation_split = 0.2,
	callbacks=[tbCallBack]
)
test_X = pd.read_csv('./test_cleaned.csv').values
whether_claim = test_X[:,3]>0

test_X = test_X[:,[5,7,10,12,13,14,15,16]]
test_X = test_X[:,:-1]
test_X = process(test_X)
test_X = test_X/max_X

y_pred = model.predict(test_X)*22025.+1035
y_pred[whether_claim>0] = 0

customer_id = np.load('./test_customer_id.npy')
test_pf = pd.DataFrame(index = customer_id,data = y_pred)
submission_data = pd.read_csv('./sample_submission.csv',index_col=0)
for i in range(len(submission_data)):
	submission_data.loc[submission_data.index[i],'claim_amount'] = test_pf[0][submission_data.index[i]]

submission_data.to_csv('./sub.csv')

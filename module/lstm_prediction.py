
# coding: utf-8

# In[11]:


# coding: utf-8
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[12]:


DataFileDataFile = "F:\\Projects\\Git\\TSP\\data\\1711.csv"
df = read_csv(DataFileDataFile, names=['date', 'crossid', 'count'])

df




df275 = df[df['crossid']==275].sort_values(by='date')
df275.index = range(1,len(df275) + 1) 


# In[15]:


dataset = df275['count'].values.astype('float32')


# In[16]:


plt.plot(dataset)
plt.show()


# In[17]:


scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset.reshape(-1, 1))


# In[18]:


#convert an array of valus to dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


# In[19]:


train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[20]:


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1 , trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



# In[21]:


trainY


# In[22]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[23]:


trainPredict = model.predict(trainX)


# In[24]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)


# In[25]:


testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)


# In[26]:


trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))


# In[27]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

#test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

plt.show()



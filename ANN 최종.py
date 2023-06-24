#open library
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler

#Define function
def IOA(mod,obs):
    ia = 1 -(np.sum((obs-mod)**2))/(np.sum((np.abs(mod-np.mean(obs))+np.abs(obs-np.mean(obs)))**2))
    global In_OA
    In_OA = ia

mima_sc = MinMaxScaler()

tf.random.set_seed(919)
file_path = 'C:/Users/Atmos/Desktop/raw.csv'

##T+72
#prepare dataset
raw_data = pd.read_csv(file_path, encoding = 'cp949')
raw_data = raw_data[((raw_data["date"] >= "2022-05-00 00:00") & (raw_data["date"] < "2022-06-00 00:00")) | (raw_data["date"] >= "2023-05-00 00:00")]
raw_data["T72"] = raw_data["T"].shift(72)
min_max = mima_sc.fit(raw_data[["T72","Wind","Pre","RH"]])
mini = mima_sc.data_min_
maxi = mima_sc.data_max_
raw_data_sc = mima_sc.transform(raw_data[["T72","Wind","Pre","RH"]])
data_sc = pd.DataFrame(raw_data_sc, columns = ["T72","Wind","Pre","RH"])

#drop NaN values in data
data_raw = data_sc.dropna(axis=0)    

#seperate data
d_train_1 = data_raw[:-72]
d_test_1 = data_raw[-72:]
#select features in train/test
d_train1_X = d_train_1[["Wind","Pre","RH"]]
d_train1_Y = d_train_1[["T72"]]
d_test1_X = d_test_1[["Wind","Pre","RH"]]
d_test1_Y = d_test_1[["T72"]]

#change shape
d_train1_X1 = d_train1_X.to_numpy().astype('float64')
d_train1_Y1 = d_train1_Y.to_numpy().astype('float64')
d_test1_X1 = d_test1_X.to_numpy().astype('float64')
d_test1_Y1 = d_test1_Y.to_numpy().astype('float64')

#final model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [3,1]),
    keras.layers.Dense(9, activation = "elu"),
    keras.layers.Dense(9, activation = "elu"),
    keras.layers.Dense(1)
    ])

model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), metrics = ['mean_squared_error'])

model.fit(d_train1_X1, d_train1_Y1, epochs = 100, validation_data = (d_train1_X1,d_train1_Y1), batch_size = 30)
final_mod_conc = model.predict(d_test1_X1)[:,0] * (maxi[0] - mini[0]) + mini[0]
final_obs_conc = d_test1_Y1[:,0] * (maxi[0] - mini[0]) + mini[0]

IOA(final_mod_conc, final_obs_conc)
final_IOA = In_OA
final_MAE = np.mean(keras.losses.mean_absolute_error(final_mod_conc,final_obs_conc))

df1 = pd.DataFrame(final_mod_conc, columns=["T"])


#RH+72
#prepare dataset
raw_data = pd.read_csv(file_path, encoding = 'cp949')
raw_data = raw_data[((raw_data["date"] >= "2022-05-15 00:00") & (raw_data["date"] < "2022-06-16 00:00")) | (raw_data["date"] >= "2023-05-00 00:00")]
raw_data["RH72"] = raw_data["RH"].shift(72)
min_max = mima_sc.fit(raw_data[["RH72","T","Wind","Pre"]])
mini = mima_sc.data_min_
maxi = mima_sc.data_max_
raw_data_sc = mima_sc.transform(raw_data[["RH72","T","Wind","Pre"]])
data_sc = pd.DataFrame(raw_data_sc, columns = ["RH72","T","Wind","Pre"])

#drop NaN values in data
data_raw = data_sc.dropna(axis=0)    

#seperate data
d_train_1 = data_raw[:-72]
d_test_1 = data_raw[-72:]
#select features in train/test
d_train1_X = d_train_1[["T","Wind","Pre"]]
d_train1_Y = d_train_1[["RH72"]]
d_test1_X = d_test_1[["T","Wind","Pre"]]
d_test1_Y = d_test_1[["RH72"]]

#change shape
d_train1_X1 = d_train1_X.to_numpy().astype('float64')
d_train1_Y1 = d_train1_Y.to_numpy().astype('float64')
d_test1_X1 = d_test1_X.to_numpy().astype('float64')
d_test1_Y1 = d_test1_Y.to_numpy().astype('float64')

#final model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [3,1]),
    keras.layers.Dense(9, activation = "elu"),
    keras.layers.Dense(9, activation = "elu"),
    keras.layers.Dense(1)
    ])

model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), metrics = ['mean_squared_error'])

model.fit(d_train1_X1, d_train1_Y1, epochs = 100, validation_data = (d_train1_X1,d_train1_Y1), batch_size = 30)
final_mod_conc = model.predict(d_test1_X1)[:,0] * (maxi[0] - mini[0]) + mini[0]
final_obs_conc = d_test1_Y1[:,0] * (maxi[0] - mini[0]) + mini[0]

IOA(final_mod_conc, final_obs_conc)
final_IOA = In_OA
final_MAE = np.mean(keras.losses.mean_absolute_error(final_mod_conc,final_obs_conc))

df2 = pd.DataFrame(final_mod_conc, columns=["RH"])


##Pre+72
#prepare dataset
raw_data = pd.read_csv(file_path, encoding = 'cp949')
raw_data = raw_data[((raw_data["date"] >= "2022-05-15 00:00") & (raw_data["date"] < "2022-06-16 00:00")) | (raw_data["date"] >= "2023-05-00 00:00")]
raw_data["Pre72"] = raw_data["Pre"].shift(72)
min_max = mima_sc.fit(raw_data[["Pre72","T","Wind","RH"]])
mini = mima_sc.data_min_
maxi = mima_sc.data_max_
raw_data_sc = mima_sc.transform(raw_data[["Pre72","T","Wind","RH"]])
data_sc = pd.DataFrame(raw_data_sc, columns = ["Pre72","T","Wind","RH"])

#drop NaN values in data
data_raw = data_sc.dropna(axis=0)    

#seperate data
d_train_1 = data_raw[:-72]
d_test_1 = data_raw[-72:]
#select features in train/test
d_train1_X = d_train_1[["T","Wind","RH"]]
d_train1_Y = d_train_1[["Pre72"]]
d_test1_X = d_test_1[["T","Wind","RH"]]
d_test1_Y = d_test_1[["Pre72"]]

#change shape
d_train1_X1 = d_train1_X.to_numpy().astype('float64')
d_train1_Y1 = d_train1_Y.to_numpy().astype('float64')
d_test1_X1 = d_test1_X.to_numpy().astype('float64')
d_test1_Y1 = d_test1_Y.to_numpy().astype('float64')

#final model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [3,1]),
    keras.layers.Dense(9, activation = "elu"),
    keras.layers.Dense(9, activation = "elu"),
    keras.layers.Dense(1)
    ])

model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), metrics = ['mean_squared_error'])

model.fit(d_train1_X1, d_train1_Y1, epochs = 100, validation_data = (d_train1_X1,d_train1_Y1), batch_size = 30)
final_mod_conc = model.predict(d_test1_X1)[:,0] * (maxi[0] - mini[0]) + mini[0]
final_obs_conc = d_test1_Y1[:,0] * (maxi[0] - mini[0]) + mini[0]

IOA(final_mod_conc, final_obs_conc)
final_IOA = In_OA
final_MAE = np.mean(keras.losses.mean_absolute_error(final_mod_conc,final_obs_conc))

df3 = pd.DataFrame(final_mod_conc, columns=["Pre"])

#final_prediction
start_date = "2023-05-24 00:00"
end_date = "2023-05-26 23:00"
date_range = pd.date_range(start=start_date, end=end_date, freq="H")
df0 = pd.DataFrame({"date": date_range})
df0["date"] = pd.to_datetime(df0["date"])
dfs = [df0,df1,df2,df3]
df = pd.concat(dfs, axis=1)

df = df[df["date"] >= "2023-05-26 00:00"]

df_t = df["T"]
df_t_max = np.max(df_t)
df_t_min = np.min(df_t)
df_t_mean = np.mean(df_t)

df_rh = df["RH"]
df_rh_max = np.max(df_rh)
df_rh_min = np.min(df_rh)
df_rh_mean = np.mean(df_rh)

df_pre = df["Pre"]
df_pre[df_pre < 0] = 0
df_pre_mean = np.mean(df_pre)
df_pre_sum = np.sum(df_pre)

print("기온 일 최대값: ", df_t_max, "°C")
print("기온 일 최소값: ", df_t_min, "°C")
print("기온 24시간 평균값: ", df_t_mean, "°C")
print("슴도 일 최대값: ", df_rh_max, "%")
print("습도 일 최소값: ", df_rh_min, "%")
print("습도 24시간 평균값: ", df_rh_mean, "%")
print("일 평균 강수량: ", df_pre_mean, "mm/h")
print("일간 총 강수량: ", df_pre_sum, "mm")

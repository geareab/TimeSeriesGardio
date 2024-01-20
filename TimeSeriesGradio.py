from catboost import CatBoostRegressor
import lightgbm
import gradio as gr


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


cat = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=3,
    random_state=100)
cat.load_model("catboost.cbm")

lgbm = lightgbm.Booster(model_file='lgbr_base.txt')


dfx = pd.read_csv('10TestHouses.csv')
dfx = dfx.drop(columns = ['LCLid', 'Acorn_grouped'])
dfx = dfx.reset_index(drop=True)
dfx.head()

dat = []
for index, row in dfx.iterrows():
    if(row['KWH/hh (per half hour) '] == 'Null'):
        row['KWH/hh (per half hour) '] = 0
    dat.append(float(row['KWH/hh (per half hour) ']))

trainset = pd.DataFrame({"KWH/hh (per half hour) " : dat})
scaler = MinMaxScaler(feature_range = (0,1))
trainset_scaled = scaler.fit_transform(trainset)

print(len(trainset_scaled))

X = []
y = []

for i in range(25, len(trainset_scaled)):
    X.append(trainset_scaled[i-25:i-1, 0])
    y.append(trainset_scaled[i-1, 0])
X, y = np.array(X), np.array(y)

X_test = np.reshape(X, (X.shape[0], X.shape[1]))
y_test = np.reshape(y, (y.shape[0], 1))

y_test = np.reshape(y_test, (y_test.shape[0]))

y_test = np.reshape(y_test, (y_test.shape[0], 1))
y_test = scaler.inverse_transform(y_test)
y_test = np.reshape(y_test, (y_test.shape[0]))

def predict(sequence_number, sequence = '', true_output = 0):
    
    if sequence == '':
        cat_y = scaler.inverse_transform(np.reshape(cat.predict(X_test[sequence_number]), (1, -1)))[0][0]
        lgbm_y = scaler.inverse_transform(np.reshape(lgbm.predict([X_test[sequence_number]]), (1, -1)))[0][0]
        real_y = y_test[sequence_number]
        return '[' + ', '.join(map(str, X_test[sequence_number])) + ']', cat_y, lgbm_y, real_y
    else:
        cat_y = scaler.inverse_transform(np.reshape(cat.predict(eval(sequence)), (1, -1)))[0][0]
        lgbm_y = scaler.inverse_transform(np.reshape(lgbm.predict([eval(sequence)]), (1, -1)))[0][0]
        real_y = true_output
        return sequence, cat_y, lgbm_y, real_y

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Slider(0, len(trainset_scaled)-100), 'text', 'number'],
    outputs=["text", "number", "number", "number"],
    title = 'Smart Grid Load Forecasting'
)

demo.launch()

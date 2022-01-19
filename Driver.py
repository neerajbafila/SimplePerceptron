from Perceptron import  Perceptron as pton
import pandas as pd

def create_data():

    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,1]
    }

    df_OR = pd.DataFrame(OR)

    return df_OR

def prepare_data(df, target_col="y"):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y

df_or = create_data()
X, y = prepare_data(df_or)
ETA = 0.1
EPOCHS = 10
model_or = pton(learning_rate=ETA, epochs=EPOCHS)

model_or.fit(X, y)

print(model_or.predict([[0,1]]))

import os
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

os.chdir(r"c:\python projects")
df = pd.read_excel("semiconductordata.xlsx")
print('hi')

# Hot-encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['crystal_system'])

# Declaring x (independent variable) and y (output variable)
x = df_encoded.drop(['band_gap', 'Compound'], axis=1)
y = df_encoded['band_gap']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=32)

#numerical columns for scaling 
numerical_columns = ['lattice_parameter_a', 'lattice_parameter_c', 'temperature']

# MinMAx Scaler
scaler = StandardScaler()

# Check if numerical columns exist in x_train before scaling
if all(col in x_train.columns for col in numerical_columns):
    x_train[numerical_columns] = scaler.fit_transform(x_train[numerical_columns])
    x_test[numerical_columns] = scaler.transform(x_test[numerical_columns])  # Transform test data
else:
    print("Some numerical columns are missing in x_train.")

# Defining sequential (fastfeed) neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),  # Input layer with number of features
    layers.Dense(32, activation='relu'),  # Hidden layer
    layers.Dense(1)  # Output layer for regression
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
test_loss = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)

# Making predictions on test data
predictions = model.predict(x_test)

#(actual band gap values)
print("y_test (Actual Band Gap Values):", y_test.values)

# Display predictions alongside actual values
results = pd.DataFrame({
    'Predicted Band Gap': predictions.flatten(),
    'Actual Band Gap': y_test.values
})

print(results)


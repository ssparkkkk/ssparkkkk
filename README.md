import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Crear datos de entrenamiento y prueba
samples = 1000

# Entradas
X = np.random.uniform(0, 10, (samples, 2))  # Generar datos aleatorios entre 0 y 10. El resultado es algo como esto: [[1, 5], [4, 2], [9, 2]...]

# Salidas
Y = X[:, 0] + X[:, 1] # Generar las salidas sumando la primera y segunda columna de los valores de X. Resultado es algo como: [6, 2, 11...]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # 0.2 = 20% datos de prueba y 80% de datos de entrenamiento

# Crear modelo de red neuronal
model = Sequential([
    Dense(16, input_dim=2, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compilar modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1) # El batch_size Define el número de muestras que se pasarán a través de la red antes de que se actualicen los parámetros del modelo (por ejemplo, los pesos)mport numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Crear datos de entrenamiento y prueba
samples = 1000

# Entradas
X = np.random.uniform(0, 10, (samples, 2))  # Generar datos aleatorios entre 0 y 10. El resultado es algo como esto: [[1, 5], [4, 2], [9, 2]...]

# Salidas
Y = X[:, 0] + X[:, 1] # Generar las salidas sumando la primera y segunda columna de los valores de X. Resultado es algo como: [6, 2, 11...]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # 0.2 = 20% datos de prueba y 80% de datos de entrenamiento

# Crear modelo de red neuronal
model = Sequential([
    Dense(16, input_dim=2, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compilar modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1) # El batch_size Define el número de muestras que se pasarán a través de la red antes de que se actualicen los parámetros del modelo (por ejemplo, los pesos)

# Realizar predicciones
input_data = np.array([[4, 4]])
result = model.predict(input_data)
print("Resultado de la suma:", round(result[0][0]))

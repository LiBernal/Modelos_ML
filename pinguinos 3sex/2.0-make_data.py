import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Cargar datos
df = pd.read_csv("penguins.csv")

# Descartar inválidos
df=df.fillna({'sex':'unknown'}) 
df_clean = df.dropna().reset_index(drop=True)

# Codificamos como 0 y 1 el sexo
encoder=LabelEncoder()
encoder.fit(['male','unknown','female'])
df_clean['sex_encoded'] = encoder.transform(df_clean['sex'])

numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'] 

# Eliminamos outliers (usando del método de 1.5 veces la distancia inter-cuartil
dd = df_clean.describe()
for col in numeric_cols:
    distancia_ic=dd[col]['75%']-dd[col]['25%']
    limite_inferior=dd[col]['25%']-1.5*distancia_ic
    limite_superior=dd[col]['75%']+1.5*distancia_ic
    df_clean=df_clean[(df_clean[col]>limite_inferior) & (df_clean[col]<limite_superior)]

features = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'sex_encoded'] #Ignoramos el tamaño de aleta
X = df_clean[features].values
y_species = df_clean['species'].values

# Normalizar
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
file.close()
# Mapear especies a números para comparación
encoder.fit(['Adelie','Chinstrap','Gentoo'])
y_numeric = encoder.transform(y_species)

x_t,x_test,y_t,y_test           = train_test_split(X_scaled,y_numeric,train_size=0.8,stratify=y_numeric)
x_train,x_valid,y_train,y_valid = train_test_split(x_t,y_t,test_size=0.125, stratify=y_t)

np.savetxt("data/X_full.csv", X_scaled, delimiter=",")
np.savetxt("data/Y_full.csv", y_numeric, delimiter=",")
np.savetxt("data/X_train.csv", x_train, delimiter=",")
np.savetxt("data/Y_train.csv", y_train, delimiter=",")
np.savetxt("data/X_valid.csv", x_valid, delimiter=",")
np.savetxt("data/Y_valid.csv", y_valid, delimiter=",")
np.savetxt("data/X_test.csv", x_test, delimiter=",")
np.savetxt("data/Y_test.csv", y_test, delimiter=",")

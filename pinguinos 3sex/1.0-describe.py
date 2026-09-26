import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Cargar y limpiar datos
df = pd.read_csv("penguins.csv")
print("*"*80)
print("Descripcion de los datos:")
print(df.describe())
print("*"*80)
print("valores nulos por columna:")
print("*"*80)
print(df.isnull().sum())
df=df.fillna({'sex':'unknown'}) 
print(df[df['bill_length_mm'].isnull()])

df_clean = df.dropna().reset_index(drop=True)

numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']


fig, axes = plt.subplots(2, 2, figsize=(10,10))
plotsdd = [('bill_length_mm',axes[0,0]), ('bill_depth_mm',axes[0,1]), ('flipper_length_mm',axes[1,0]),('body_mass_g',axes[1,1])]
for col,axis in plotsdd:
    df_clean[[col]].boxplot(ax=axis)
plt.show()  

print("*"*80)
print("Sesgo(skew) y Curtosis")
print("*"*80)

for col in numeric_cols:
    print("Variable:",col,"Sesgo:",df[col].skew(),"Curtosis:",df[col].kurt())

print("*"*80)
print("Prueba de Shapiro de normalidad")
print("*"*80)

for col in numeric_cols:
    mean= df_clean[col].mean()
    std = df_clean[col].std()
    stat, p_value = stats.shapiro(df_clean[col])
    print(col, stat, p_value)

for col in numeric_cols:
    mean = df_clean[col].mean()
    std = df_clean[col].std()
    sns.histplot(df_clean[col], kde=True)
    plt.axvline(mean, color='red', linestyle='--', label='Media')
    plt.axvline(mean + std, color='green', linestyle='--', label='Desviación Estándar')
    plt.axvline(mean - std, color='green', linestyle='--')
    plt.title(f'Histograma de {col} con Media y Desviación Estándar')
    plt.legend()
    plt.show()

dd = df_clean.describe()

for col in numeric_cols:
    distancia_ic=dd[col]['75%']-dd[col]['25%']
    limite_inferior=dd[col]['25%']-1.5*distancia_ic
    limite_superior=dd[col]['75%']+1.5*distancia_ic
    df_clean=df_clean[(df_clean[col]>limite_inferior) & (df_clean[col]<limite_superior)]

# 0.0 Get dataset
###  pip install --upgrade palmerpenguins ###
from palmerpenguins import load_penguins
df = load_penguins()
df.to_csv("penguins.csv")

# 1.0 Preprocesamiento
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

# 1.1 Compare boxes

# Cargar y limpiar datos
df = pd.read_csv("penguins.csv")
df_clean = df.dropna().reset_index(drop=True)

numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# 5. Boxplot comparativo 
df_melted = df_clean[['bill_length_mm'] + ['species']].melt(
    id_vars=['species'], 
    var_name='bill_length_mm', 
    value_name='Longitud (pico) [mm]'
)
fig, axes = plt.subplots(2, 2, figsize=(10,10))
sns.boxplot(data=df_melted, x='bill_length_mm', y='Longitud (pico) [mm]', hue='species', ax=axes[0,0])

df_melted = df_clean[['bill_depth_mm'] + ['species']].melt(
    id_vars=['species'], 
    var_name='bill_depth_mm', 
    value_name='Profundidad (pico) [mm]'
)
sns.boxplot(data=df_melted, x='bill_depth_mm', y='Profundidad (pico) [mm]', hue='species', ax=axes[0,1])

df_melted = df_clean[['flipper_length_mm'] + ['species']].melt(
    id_vars=['species'], 
    var_name='flipper_length_mm', 
    value_name='Longitud (aleta) [mm]'
)
sns.boxplot(data=df_melted, x='flipper_length_mm', y='Longitud (aleta) [mm]', hue='species', ax=axes[1,0])

df_melted = df_clean[['body_mass_g'] + ['species']].melt(
    id_vars=['species'], 
    var_name='body_mass_g', 
    value_name='Masa corporal [g]'
)
sns.boxplot(data=df_melted, x='body_mass_g', y='Masa corporal [g]', hue='species', ax=axes[1,1])
fig.suptitle('Todas las Variables por Especie', fontsize=14, fontweight='bold')
fig.supxlabel('Variable', fontsize=12)
plt.legend(title='Species', loc='upper right')
plt.tight_layout()
plt.show()

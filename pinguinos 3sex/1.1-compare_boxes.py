import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

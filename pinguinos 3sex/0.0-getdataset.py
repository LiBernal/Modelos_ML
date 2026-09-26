###  pip install --upgrade palmerpenguins ###
from palmerpenguins import load_penguins
df = load_penguins()
df.to_csv("penguins.csv")

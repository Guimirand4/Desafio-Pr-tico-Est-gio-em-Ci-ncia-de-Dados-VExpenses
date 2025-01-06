import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar o Dataset
df = pd.read_csv('Dataset.csv')

# 2. Análise Exploratória
print(df.info())
print(df.describe())
sns.pairplot(df, hue='Compra (0 ou 1)')

# 3. Pré-processamento
df['Anúncio Clicado'] = df['Anúncio Clicado'].map({'Sim': 1, 'Não': 0})
df['Gênero'] = LabelEncoder().fit_transform(df['Gênero'])

X = df.drop('Compra (0 ou 1)', axis=1)
y = df['Compra (0 ou 1)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Construção do Modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Avaliação
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()





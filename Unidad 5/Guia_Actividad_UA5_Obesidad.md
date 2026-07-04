# Guía para desarrollar la actividad UA5 - Aprendizaje Supervisado
### Dataset: obesity_dataset

---

## 1. Primero, entiende qué tipo de problema es

Tu variable objetivo es **`NObeyesdad`** (nivel de obesidad), que tiene 7 categorías:
`Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III`

Como es una variable **categórica** (no un número continuo), esto es un problema de **CLASIFICACIÓN**, no de regresión.

👉 Esto significa que tu referencia principal debe ser el **Caso 12 (churn)**, que también es clasificación. El Caso 11 (arriendos) te sirve sobre todo para técnicas de EDA y preprocesamiento, porque ese caso es de regresión (predecía un precio).

*Nota: Si quisieras, también podrías plantear un problema de regresión prediciendo `Weight` (peso) a partir de las demás variables — ahí sí usarías el Caso 11 como referencia principal. Pero lo más natural dado el dataset es clasificar el nivel de obesidad.*

---

## 2. Estructura sugerida del notebook (siguiendo la rúbrica)

La rúbrica pide 4 cosas — tu notebook debe tener estas secciones claramente:

1. **EDA y preparación de datos** (1.25 pts)
2. **Selección e implementación del modelo** (1.25 pts)
3. **Evaluación y validación del modelo** (1.25 pts)
4. **Interpretación de resultados y conclusiones** (1.25 pts)

A continuación te explico qué hacer en cada una, con el código como en los casos vistos.

---

## 3. Librerías necesarias

Igual que en el Caso 12, pero recuerda que aquí tienes que codificar **muchas más variables categóricas**:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
```

---

## 4. EDA (Análisis Exploratorio) — Criterio 1

Tu dataset **no tiene datos faltantes** (ya lo revisé), así que no necesitas hacer `dropna()`. Aun así, debes mostrar que lo verificaste:

```python
df = pd.read_csv('obesity_dataset.csv')
df.head()
df.info()
df.describe()          # estadísticas de variables numéricas
df.isna().sum()        # confirmar que no hay nulos
df['NObeyesdad'].value_counts()   # ver balance de clases
```

**Variables numéricas:** `Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE`
**Variables categóricas:** `Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS`

### Visualizaciones recomendadas (como en el Caso 12)

Para que el EDA sea "exhaustivo" (nivel Excelente en la rúbrica), no te quedes solo con histogramas sueltos — relaciona las variables con el target:

```python
# Distribución de una variable numérica según el nivel de obesidad
plt.figure(figsize=(10, 5))
sns.boxplot(x='NObeyesdad', y='Weight', data=df)
plt.xticks(rotation=45)
plt.title('Peso según nivel de obesidad')
plt.show()

# Variable categórica vs target
plt.figure(figsize=(8, 5))
sns.countplot(x='family_history_with_overweight', hue='NObeyesdad', data=df)
plt.title('Historial familiar de sobrepeso vs nivel de obesidad')
plt.show()

# Matriz de correlación entre numéricas
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlación entre variables numéricas')
plt.show()
```

Otras ideas: `Height` vs `Weight` con color por `NObeyesdad` (scatterplot), o `FAF` (actividad física) vs el nivel de obesidad.

---

## 5. Preprocesamiento — sigue siendo parte del Criterio 1

Aquí es donde el dataset de obesidad es un poco más exigente que los casos vistos, porque tiene **muchas variables categóricas de distinto tipo**:

### a) Variables binarias (yes/no) → conviértelas a 0/1

```python
binarias = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binarias:
    df[col] = df[col].map({'yes': 1, 'no': 0})
```

### b) Variables ordinales (tienen un orden natural: no < a veces < frecuente < siempre)

`CAEC` y `CALC` tienen las categorías: `no, Sometimes, Frequently, Always`. Como sí tienen un orden lógico, puedes mapearlas a números en vez de usar OneHotEncoder:

```python
orden = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC'] = df['CAEC'].map(orden)
df['CALC'] = df['CALC'].map(orden)
```

### c) Variables nominales (sin orden: Gender, MTRANS) → OneHotEncoder o `pd.get_dummies`

```python
df = pd.get_dummies(df, columns=['Gender', 'MTRANS'], drop_first=True)
```

### d) La variable objetivo (target) también hay que codificarla

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['NObeyesdad_cod'] = le.fit_transform(df['NObeyesdad'])
# le.classes_ te muestra el orden: qué número corresponde a cada categoría
```

### e) Separar X (predictoras) y y (target), y dividir train/test

Como en el Caso 12, usa `stratify=y` porque hay varias clases y quieres mantener las proporciones:

```python
X = df.drop(columns=['NObeyesdad', 'NObeyesdad_cod'])
y = df['NObeyesdad_cod']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### f) Escalado (necesario para KNN y SVM, no tan crítico para árboles)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 6. Selección e implementación del modelo — Criterio 2

La guía de actividad menciona explícitamente **SVM, Random Forest y KNN**. Como es clasificación multiclase (7 categorías), te recomiendo probar al menos 2-3 modelos y comparar, tal como en el Caso 12:

```python
# Random Forest - buena opción como modelo base, no requiere escalado
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# KNN - requiere datos escalados
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

# SVM - también requiere datos escalados
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
```

**Justificación que puedes usar (adáptala a tus resultados del EDA):** Random Forest suele funcionar bien aquí porque hay variables categóricas codificadas y relaciones no necesariamente lineales entre las variables y el nivel de obesidad (ej. el peso y la altura no se relacionan linealmente con las 7 categorías). KNN y SVM son buenos puntos de comparación porque dependen de distancias entre observaciones.

*Opcional (nivel "Excelente" en la rúbrica): usa `GridSearchCV` para afinar hiperparámetros, igual que en el Caso 11 con el número de vecinos de KNN.*

```python
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Mejores parámetros:", grid.best_params_)
```

---

## 7. Evaluación y validación — Criterio 3

Como en el Caso 12, usa matriz de confusión, `classification_report` (que ya incluye precisión, recall y F1) y accuracy. Como aquí hay 7 clases (no 2 como en churn), **no apliques ROC-AUC directamente** — esa métrica es más simple de interpretar en clasificación binaria. Concéntrate en:

```python
for nombre, y_pred in [('Random Forest', y_pred_rf), ('KNN', y_pred_knn), ('SVM', y_pred_svm)]:
    print(f'--- {nombre} ---')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print()

# Matriz de confusión (elige el mejor modelo, ej. Random Forest)
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Random Forest')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
```

---

## 8. Interpretación de resultados y conclusiones — Criterio 4

Aquí no basta con reportar los números — debes explicar qué significan. Ideas para guiarte:

- **Feature importance** (igual que en el Caso 12 con Gradient Boosting): muestra qué variables pesan más en la predicción.

```python
importancias = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importancia', y='Variable', data=importancias)
plt.title('Importancia de variables - Random Forest')
plt.show()
```

- Compara los 3 modelos: ¿cuál tuvo mejor accuracy y F1? ¿Hubo alguna categoría (ej. `Obesity_Type_III`) que el modelo predijo mejor o peor que otras? Revisa la matriz de confusión para eso.
- Relaciona con el EDA: si `Weight` y `family_history_with_overweight` salieron como las variables más importantes, coméntalo — tiene sentido con lo que viste al inicio.
- Cierra con una conclusión breve: ¿el modelo es útil en la práctica? ¿qué limitaciones tiene (ej. clases con pocos datos, variables autoreportadas como hábitos alimenticios)?

---

## 9. Checklist antes de entregar

- [ ] EDA con al menos 3-4 visualizaciones relacionando variables con el target
- [ ] Verificación de nulos (aunque no haya)
- [ ] Todas las variables categóricas codificadas (binarias, ordinales, nominales)
- [ ] División train/test con `stratify`
- [ ] Al menos 2 modelos de clasificación implementados y comparados
- [ ] Métricas: accuracy, classification_report (precision/recall/F1), matriz de confusión
- [ ] Interpretación de resultados con feature importance
- [ ] Conclusiones que respondan: ¿qué tan bien predice el modelo y por qué?

---

*Guía elaborada a partir de la Guía de Actividades UA5 y los notebooks de referencia Caso 11 (regresión) y Caso 12 (clasificación).*

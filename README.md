# Health Insurance Cross Sell Prediction Research Notebook

## Описание проекта

Этот блокнот содержит полное исследование задачи бинарной классификации. Мы предсказываем реакцию клиентов на предложение мед страхования. Задача взята из соревнования Kaggle Playground Series Season 4 Episode 7.

### Структура исследования:

1. **EDA (Exploratory Data Analysis)**: анализ данных
2. **LightAutoML Baseline**: базовое решение с использованием LightAutoML, две конфигурации
3. **Custom Solution**: собственные решения без использования LightAutoML
   * Простые pipeline подходы
   * Улучшенные методы с очисткой данных
   * CatBoost решения: простой, продвинутый, с Optuna
4. **Выводы и суммаризация**: анализ результатов и заключение

### Метрика: ROC-AUC

## 1. Импорт библиотек и настройка окружения


```python
import os
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import optuna
import joblib
from scipy import stats

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

RANDOM_STATE = 42
N_THREADS = 16
N_FOLDS = 5
TEST_SIZE = 0.2
TIMEOUT = 1800

DATA_DIR = Path('../playground-series-s4e7')
if not DATA_DIR.exists():
    DATA_DIR = Path('playground-series-s4e7')
MODELS_DIR = Path('src/models')
MODELS_DIR.mkdir(exist_ok=True)

PARAMS_DIR = Path('src/params')
PARAMS_DIR.mkdir(exist_ok=True)

SUBMIT_DIR = Path('src/submission')
SUBMIT_DIR.mkdir(exist_ok=True)

OTHER_DIR = Path('src/other')
OTHER_DIR.mkdir(exist_ok=True)

print("Библиотеки импортированы")
print(f"Директория данных: {DATA_DIR.absolute()}")
print(f"Директория моделей: {MODELS_DIR.absolute()}")
print(f"Директория параметров: {PARAMS_DIR.absolute()}")
print(f"Директория submission: {SUBMIT_DIR.absolute()}")
print(f"Директория других артефактов: {OTHER_DIR.absolute()}")
```

## 2. Загрузка данных


```python
train_data = pd.read_csv(DATA_DIR / 'train.csv')
test_data = pd.read_csv(DATA_DIR / 'test.csv')
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')

print(f"Размер обучающей выборки: {train_data.shape}")
print(f"Размер тестовой выборки: {test_data.shape}")
print(f"\nКолонки в данных:")
print(train_data.columns.tolist())
print(f"\nПервые строки обучающей выборки:")
train_data.head()
```

    Размер обучающей выборки: (11504798, 12)
    Размер тестовой выборки: (7669866, 11)
    
    Колонки в данных:
    ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response']
    
    Первые строки обучающей выборки:
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Male</td>
      <td>21</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>65101.0</td>
      <td>124.0</td>
      <td>187</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>43</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>58911.0</td>
      <td>26.0</td>
      <td>288</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Female</td>
      <td>25</td>
      <td>1</td>
      <td>14.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>38043.0</td>
      <td>152.0</td>
      <td>254</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Female</td>
      <td>35</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>2630.0</td>
      <td>156.0</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Female</td>
      <td>36</td>
      <td>1</td>
      <td>15.0</td>
      <td>1</td>
      <td>1-2 Year</td>
      <td>No</td>
      <td>31951.0</td>
      <td>152.0</td>
      <td>294</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("данные:")
print(train_data.info())
print("\n" + "="*50)
print("\nОписательная статистика:")
print(train_data.describe())
print("\n" + "="*50)
print("\nПропущенные значения:")
missing = train_data.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Пропущенных значений не обнаружено")
```

    данные:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11504798 entries, 0 to 11504797
    Data columns (total 12 columns):
     #   Column                Dtype  
    ---  ------                -----  
     0   id                    int64  
     1   Gender                object 
     2   Age                   int64  
     3   Driving_License       int64  
     4   Region_Code           float64
     5   Previously_Insured    int64  
     6   Vehicle_Age           object 
     7   Vehicle_Damage        object 
     8   Annual_Premium        float64
     9   Policy_Sales_Channel  float64
     10  Vintage               int64  
     11  Response              int64  
    dtypes: float64(3), int64(6), object(3)
    memory usage: 1.0+ GB
    None
    
    ==================================================
    
    Описательная статистика:
                     id           Age  Driving_License   Region_Code  \
    count  1.150480e+07  1.150480e+07     1.150480e+07  1.150480e+07   
    mean   5.752398e+06  3.838356e+01     9.980220e-01  2.641869e+01   
    std    3.321149e+06  1.499346e+01     4.443120e-02  1.299159e+01   
    min    0.000000e+00  2.000000e+01     0.000000e+00  0.000000e+00   
    25%    2.876199e+06  2.400000e+01     1.000000e+00  1.500000e+01   
    50%    5.752398e+06  3.600000e+01     1.000000e+00  2.800000e+01   
    75%    8.628598e+06  4.900000e+01     1.000000e+00  3.500000e+01   
    max    1.150480e+07  8.500000e+01     1.000000e+00  5.200000e+01   
    
           Previously_Insured  Annual_Premium  Policy_Sales_Channel       Vintage  \
    count        1.150480e+07    1.150480e+07          1.150480e+07  1.150480e+07   
    mean         4.629966e-01    3.046137e+04          1.124254e+02  1.638977e+02   
    std          4.986289e-01    1.645475e+04          5.403571e+01  7.997953e+01   
    min          0.000000e+00    2.630000e+03          1.000000e+00  1.000000e+01   
    25%          0.000000e+00    2.527700e+04          2.900000e+01  9.900000e+01   
    50%          0.000000e+00    3.182400e+04          1.510000e+02  1.660000e+02   
    75%          1.000000e+00    3.945100e+04          1.520000e+02  2.320000e+02   
    max          1.000000e+00    5.401650e+05          1.630000e+02  2.990000e+02   
    
               Response  
    count  1.150480e+07  
    mean   1.229973e-01  
    std    3.284341e-01  
    min    0.000000e+00  
    25%    0.000000e+00  
    50%    0.000000e+00  
    75%    0.000000e+00  
    max    1.000000e+00  
    
    ==================================================
    
    Пропущенные значения:
    Пропущенных значений не обнаружено
    

## 3. Exploratory Data Analysis (EDA)

В этом разделе мы изучаем данные. Смотрим на структуру данных, распределения признаков и целевой переменной. Ищем закономерности и связи между признаками.

### 3.1. Анализ целевой переменной

Анализируем распределение целевой переменной Response. В данных наблюдается сильный дисбаланс классов: класс 0 составляет 87.70% (10,089,739 наблюдений), а класс 1 только 12.30% (1,415,059 наблюдений). Соотношение классов составляет примерно 7.13:1. Такой дисбаланс нужно учитывать при обучении моделей, используя специальные техники балансировки классов или метрики оценки.


```python
target_col = 'Response'
target_counts = train_data[target_col].value_counts()
target_props = train_data[target_col].value_counts(normalize=True)

print("Распределение целевой переменной:")
print(f"Класс 0: {target_counts[0]} ({target_props[0]*100:.2f}%)")
print(f"Класс 1: {target_counts[1]} ({target_props[1]*100:.2f}%)")
print(f"\nОбщее количество наблюдений: {len(train_data)}")
print(f"Дисбаланс классов: {target_counts[0] / target_counts[1]:.2f}:1")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(target_counts.index, target_counts.values, color=['skyblue', 'salmon'])
axes[0].set_xlabel('Response')
axes[0].set_ylabel('Количество')
axes[0].set_title('Распределение целевой переменной (абсолютные значения)')
axes[0].set_xticks([0, 1])
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v, str(v), ha='center', va='bottom', fontsize=12)

axes[1].pie(target_counts.values, labels=['No Response (0)', 'Response (1)'], 
            autopct='%1.2f%%', colors=['skyblue', 'salmon'], startangle=90)
axes[1].set_title('Распределение целевой переменной (проценты)')

plt.tight_layout()
plt.savefig(OTHER_DIR / 'target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

    Распределение целевой переменной:
    Класс 0: 10089739 (87.70%)
    Класс 1: 1415059 (12.30%)
    
    Общее количество наблюдений: 11504798
    Дисбаланс классов: 7.13:1
    


    
![png](readme_files/readme_8_1.png)
    


### 3.2. Типизация признаков и их распределения

Делим признаки на численные и категориальные. В данных 7 численных признаков (Age, Driving_License, Region_Code, Previously_Insured, Annual_Premium, Policy_Sales_Channel, Vintage) и 3 категориальных признака (Gender, Vehicle_Age, Vehicle_Damage). Возраст клиентов варьируется от 20 до 85 лет со средним значением 38.4 года.


```python
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()

if 'id' in numeric_cols:
    numeric_cols.remove('id')
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

print("Численные признаки:", numeric_cols)
print(f"\nКоличество численных признаков: {len(numeric_cols)}")
print("\nКатегориальные признаки:", categorical_cols)
print(f"Количество категориальных признаков: {len(categorical_cols)}")


```

    Численные признаки: ['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
    
    Количество численных признаков: 7
    
    Категориальные признаки: ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
    Количество категориальных признаков: 3
    


```python
print("Статистика по числовым признакам:")
print(train_data[numeric_cols].describe())
```

    Статистика по числовым признакам:
                    Age  Driving_License   Region_Code  Previously_Insured  \
    count  1.150480e+07     1.150480e+07  1.150480e+07        1.150480e+07   
    mean   3.838356e+01     9.980220e-01  2.641869e+01        4.629966e-01   
    std    1.499346e+01     4.443120e-02  1.299159e+01        4.986289e-01   
    min    2.000000e+01     0.000000e+00  0.000000e+00        0.000000e+00   
    25%    2.400000e+01     1.000000e+00  1.500000e+01        0.000000e+00   
    50%    3.600000e+01     1.000000e+00  2.800000e+01        0.000000e+00   
    75%    4.900000e+01     1.000000e+00  3.500000e+01        1.000000e+00   
    max    8.500000e+01     1.000000e+00  5.200000e+01        1.000000e+00   
    
           Annual_Premium  Policy_Sales_Channel       Vintage  
    count    1.150480e+07          1.150480e+07  1.150480e+07  
    mean     3.046137e+04          1.124254e+02  1.638977e+02  
    std      1.645475e+04          5.403571e+01  7.997953e+01  
    min      2.630000e+03          1.000000e+00  1.000000e+01  
    25%      2.527700e+04          2.900000e+01  9.900000e+01  
    50%      3.182400e+04          1.510000e+02  1.660000e+02  
    75%      3.945100e+04          1.520000e+02  2.320000e+02  
    max      5.401650e+05          1.630000e+02  2.990000e+02  
    


```python
n_numeric = len(numeric_cols)
n_cols = 3
n_rows = (n_numeric + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(train_data[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Распределение {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Частота')
    axes[i].grid(True, alpha=0.3)

for i in range(n_numeric, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(OTHER_DIR / 'numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](readme_files/readme_12_0.png)
    


### 3.3. Выявление аномальных значений

Ищем выбросы в численных признаках. Это нужно чтобы понять качество данных и решить нужно ли их обрабатывать. Анализ показал наличие выбросов в признаке Annual_Premium: обнаружено 2,377,273 выброса (20.66% от всех данных). В признаке Driving_License найдено 22,757 выбросов (0.20%). Остальные численные признаки не содержат значимых выбросов. Выбросы в Annual_Premium могут влиять на качество модели и требуют специальной обработки.


```python
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("Анализ выбросов (IQR метод):")
print("="*60)
outliers_summary = {}

for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(train_data, col)
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(train_data)) * 100
    outliers_summary[col] = {
        'count': n_outliers,
        'percentage': pct_outliers,
        'lower_bound': lower,
        'upper_bound': upper
    }
    print(f"\n{col}:")
    print(f"Выбросов: {n_outliers} ({pct_outliers:.2f}%)")
    print(f"Границы: [{lower:.2f}, {upper:.2f}]")

n_numeric = len(numeric_cols)
n_cols = 2
n_rows = (n_numeric + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].boxplot(train_data[col].dropna(), vert=True)
    axes[i].set_title(f'Boxplot для {col}')
    axes[i].set_ylabel(col)
    axes[i].grid(True, alpha=0.3)

for i in range(n_numeric, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(OTHER_DIR / 'outliers_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
```

    Анализ выбросов (IQR метод):
    ============================================================
    
    Age:
    Выбросов: 0 (0.00%)
    Границы: [-13.50, 86.50]
    
    Driving_License:
    Выбросов: 22757 (0.20%)
    Границы: [1.00, 1.00]
    
    Region_Code:
    Выбросов: 0 (0.00%)
    Границы: [-15.00, 65.00]
    
    Previously_Insured:
    Выбросов: 0 (0.00%)
    Границы: [-1.50, 2.50]
    
    Annual_Premium:
    Выбросов: 2377273 (20.66%)
    Границы: [4016.00, 60712.00]
    
    Policy_Sales_Channel:
    Выбросов: 0 (0.00%)
    Границы: [-155.50, 336.50]
    
    Vintage:
    Выбросов: 0 (0.00%)
    Границы: [-100.50, 431.50]
    


    
![png](readme_files/readme_14_1.png)
    


### 3.4. Анализ пропущенных значений

Проверяем наличие пропущенных значений в данных. Анализ показал, что во всех признаках отсутствуют пропущенные значения.


```python
missing_data = train_data.isnull().sum()
print(missing_data)

```

    id                      0
    Gender                  0
    Age                     0
    Driving_License         0
    Region_Code             0
    Previously_Insured      0
    Vehicle_Age             0
    Vehicle_Damage          0
    Annual_Premium          0
    Policy_Sales_Channel    0
    Vintage                 0
    Response                0
    dtype: int64
    

### 3.5. Определение важности признаков

Смотрим какие признаки важны. Для этого считаем корреляции с целевой переменной. Это помогает понять какие признаки дают больше всего информации. Наиболее сильная отрицательная корреляция наблюдается у признака Previously_Insured (-0.3459), что означает что клиенты которые уже застрахованы реже соглашаются на новое предложение. Признак Policy_Sales_Channel также показывает отрицательную корреляцию (-0.1527). Положительная корреляция наблюдается у признака Age (0.1221), что говорит о том что с возрастом вероятность положительного отклика немного увеличивается. Остальные признаки имеют слабую корреляцию с целевой переменной.


```python
correlations = train_data[numeric_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
target_correlations = correlations.drop(target_col)

plt.figure(figsize=(10, 6))
colors = ['red' if x < 0 else 'green' for x in target_correlations.values]
plt.barh(target_correlations.index, target_correlations.values, color=colors, alpha=0.7)
plt.xlabel('Корреляция с Response')
plt.title('Корреляции признаков с целевой переменной')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OTHER_DIR / 'feature_importance_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

top_features = target_correlations.abs().sort_values(ascending=False).head(10)
for idx, (feature, corr) in enumerate(top_features.items(), 1):
    print(f"{idx}. {feature}: {target_correlations[feature]:.4f}")
```


    
![png](readme_files/readme_18_0.png)
    


    1. Previously_Insured: -0.3459
    2. Policy_Sales_Channel: -0.1527
    3. Age: 0.1221
    4. Annual_Premium: 0.0323
    5. Vintage: -0.0152
    6. Region_Code: 0.0128
    7. Driving_License: 0.0092
    

### 3.6. Анализ возможных преобразований и генерации новых признаков

Смотрим на распределения признаков. Решаем нужно ли их преобразовывать и можно ли создать новые признаки. Признак Annual_Premium имеет скошенное распределение с большим разбросом значений (от 2,630 до 540,165), поэтому желательно логарифмическое преобразование. Возраст можно категоризировать на группы (молодые, средние, пожилые). Также можно создать признаки взаимодействия между важными признаками, например Age * Annual_Premium или комбинации категориальных признаков Vehicle_Age и Vehicle_Damage.


```python
print("1. Age")
print(f"Min: {train_data['Age'].min()}")
print(f"Max: {train_data['Age'].max()}")
print(f"Mean: {train_data['Age'].mean():.2f}\n")

print("2. Annual_Premium")
print(f"Min: {train_data['Annual_Premium'].min()}")
print(f"Max: {train_data['Annual_Premium'].max()}")
print(f"Mean: {train_data['Annual_Premium'].mean():.2f}")
```

    1. Age
    Min: 20
    Max: 85
    Mean: 38.38
    
    2. Annual_Premium
    Min: 2630.0
    Max: 540165.0
    Mean: 30461.37
    

### 3.7. Анализ зависимостей между признаками

Изучаем корреляции между признаками и их связь с целевой переменной для понимания важности признаков. Анализ корреляций показывает что большинство признаков имеют слабую связь между собой, что хорошо для обучения. Наиболее важными для предсказания являются признаки Previously_Insured, Policy_Sales_Channel и Age, которые показывают наибольшую корреляцию с целевой переменной Response.


```python
correlation_matrix = train_data[numeric_cols + [target_col]].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Корреляционная матрица численных признаков')
plt.tight_layout()
plt.savefig(OTHER_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

target_correlations = correlation_matrix[target_col].drop(target_col).sort_values(ascending=False)
print("\nкорреляции с целевой переменной:")
print("="*60)
print(target_correlations)
```


    
![png](readme_files/readme_22_0.png)
    


    
    корреляции с целевой переменной:
    ============================================================
    Age                     0.122134
    Annual_Premium          0.032261
    Region_Code             0.012816
    Driving_License         0.009197
    Vintage                -0.015177
    Policy_Sales_Channel   -0.152733
    Previously_Insured     -0.345930
    Name: Response, dtype: float64
    

### 3.8. Основные выводы EDA

После анализа данных мы выяснили следующее:

1. **Дисбаланс классов**: Классы распределены неравномерно - класс 0 составляет 87.70% (10,089,739 наблюдений), класс 1 только 12.30% (1,415,059 наблюдений). Соотношение 7.13:1. Нужно учесть это при обучении моделей используя техники балансировки классов.

2. **Важные признаки**: Наиболее важными признаками являются Previously_Insured (корреляция -0.3459), Policy_Sales_Channel (-0.1527) и Age (0.1221). Эти признаки сильнее всего связаны с целевой переменной.

3. **Типы данных**: В данных 7 численных признаков и 3 категориальных.

4. **Распределения**: Признак Annual_Premium имеет скошенное распределение с большим разбросом значений, необходимо логарифмическое преобразование.

5. **Выбросы**: Найдены выбросы в Annual_Premium (20.66% данных) и Driving_License (0.20%). Выбросы в Annual_Premium требуют специальной обработки.

6. **Пропущенные значения**: Пропущенных значений в данных нет, все 11,504,798 наблюдений содержат полную информацию.

7. **Преобразования**: Желательно логарифмирование Annual_Premium, категоризация Age, создание признаков взаимодействия между важными признаками.

## 4. Подготовка данных для моделирования

Делим данные на train и validation со стратификацией, потому что классы несбалансированы и нам важно сохранить их доли. Берем 20 процентов под validation и фиксируем random_state=42 для воспроизводимости. Утечку закрываем тем, что id не используем как признак и не добавляем информацию, которая появляется после момента предсказания.


```python
X = train_data.drop([target_col, 'id'], axis=1)
y = train_data[target_col]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер валидационной выборки: {X_val.shape}")
print(f"\nРаспределение классов в train:")
print(y_train.value_counts(normalize=True))
print(f"\nРаспределение классов в validation:")
print(y_val.value_counts(normalize=True))
```

    Размер обучающей выборки: (9203838, 10)
    Размер валидационной выборки: (2300960, 10)
    
    Распределение классов в train:
    Response
    0    0.877003
    1    0.122997
    Name: proportion, dtype: float64
    
    Распределение классов в validation:
    Response
    0    0.877003
    1    0.122997
    Name: proportion, dtype: float64
    

## 5. LightAutoML Baseline

В этом разделе делаем базовое решение с помощью LightAutoML. Проводим два эксперимента с разными настройками чтобы выбрать лучший вариант.

### 5.1. Конфигурация 1: Базовая настройка

Сначала построим самый простой бейзлайн на LightAutoML без серьезной настройки, ограничиваем время обучения и учимся на выборке 500к строк, потому что полный датасет очень большой и так быстрее проверить идею. Для надежной оценки используем кросс в 5 фолдов и фиксируем random_state.


```python
task = Task('binary', metric='auc')

roles = {
    'target': target_col,
    'drop': ['id']
}

train_lam = pd.concat([X_train, y_train], axis=1)
train_lam = train_lam.reset_index(drop=True)

sample_size = 500000
if len(train_lam) > sample_size:
    train_lam_sample = train_lam.sample(n=sample_size, random_state=RANDOM_STATE)
    print(f"Используется выборка из {sample_size} строк для обучения")
else:
    train_lam_sample = train_lam

print("Обучение LightAutoML конфигурация 1")
start_time = time.time()

automl1 = TabularAutoML(
    task=task,
    timeout=TIMEOUT,
    cpu_limit=N_THREADS,
    reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
    general_params={'use_algos': [['linear_l2', 'lgb', 'lgb_tuned', 'cb']]}
)

oof_pred1 = automl1.fit_predict(train_lam_sample, roles=roles, verbose=1)

train_time1 = time.time() - start_time
print(f"Обучение завершено за {train_time1:.2f} секунд")

val_pred1 = automl1.predict(X_val.reset_index(drop=True))
val_pred1_proba = val_pred1.data[:, 0] if val_pred1.data.ndim > 1 else val_pred1.data

score1 = roc_auc_score(y_val.reset_index(drop=True), val_pred1_proba)
print(f"ROC-AUC конфигурация 1: {score1:.6f}")

joblib.dump(automl1, MODELS_DIR / 'lama_config1.pkl')
print(f"Модель сохранена: {MODELS_DIR / 'lama_config1.pkl'}")
```

    Используется выборка из 500000 строк для обучения
    Обучение LightAutoML конфигурация 1
    [21:20:16] Stdout logging level is INFO.
    [21:20:16] Copying TaskTimer may affect the parent PipelineTimer, so copy will create new unlimited TaskTimer
    [21:20:16] Task: binary
    
    [21:20:16] Start automl preset with listed constraints:
    [21:20:16] - time: 1800.00 seconds
    [21:20:16] - CPU: 16 cores
    [21:20:16] - memory: 16 GB

    Optimization Progress: 100%|██████████| 101/101 [03:56<00:00,  2.34s/it, best_trial=100, best_value=0.871]
    
    [21:26:34] Model description:
    Final prediction for new objects (level 0) = 
    	 0.60812 * (5 averaged models Lvl_0_Pipe_0_Mod_0_LinearL2) +
    	 0.11602 * (5 averaged models Lvl_0_Pipe_1_Mod_0_LightGBM) +
    	 0.27586 * (5 averaged models Lvl_0_Pipe_1_Mod_1_Tuned_LightGBM) 
    
    Обучение завершено за 378.23 секунд
    ROC-AUC конфигурация 1: 0.875940
    Модель сохранена: src\models\lama_config1.pkl
    

### 5.2. Конфигурация 2: Расширенная настройка

Во второй конфигурации просим LightAutoML попробовать конкретные модели, если ограничить выбор алгоритмов и увеличить время обучения качество может вырасти. Это сравнение помогает понять, есть ли смысл усложнять настройку.


```python
print("Обучение LightAutoML конфигурация 2")
start_time = time.time()

automl2 = TabularAutoML(
    task=task,
    timeout=TIMEOUT * 2,
    cpu_limit=N_THREADS,
    reader_params={
        'n_jobs': N_THREADS, 
        'cv': N_FOLDS, 
        'random_state': RANDOM_STATE
    },
    general_params={
        'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]
    },
    tuning_params={
        'max_tuning_iter': 250,
        'max_tuning_time': 800,
        'fit_on_holdout': True
    },
    selection_params={
        'mode': 1,
        'importance_type': 'gain',
        'fit_on_holdout': True
    }
)
oof_pred2 = automl2.fit_predict(train_lam_sample, roles=roles, verbose=1)

train_time2 = time.time() - start_time
print(f"Обучение завершено за {train_time2:.2f} секунд")

val_pred2 = automl2.predict(X_val.reset_index(drop=True))
val_pred2_proba = val_pred2.data[:, 0] if val_pred2.data.ndim > 1 else val_pred2.data

score2 = roc_auc_score(y_val.reset_index(drop=True), val_pred2_proba)
print(f"ROC-AUC конфигурация 2: {score2:.6f}")

joblib.dump(automl2, MODELS_DIR / 'lama_config2.pkl')
print(f"Модель сохранена: {MODELS_DIR / 'lama_config2.pkl'}")
```

    Обучение LightAutoML конфигурация 2
    [11:06:04] Stdout logging level is INFO.
    [11:06:04] Task: binary
    
    [11:06:04] Start automl preset with listed constraints:
    [11:06:04] - time: 3600.00 seconds
    [11:06:04] - CPU: 16 cores
    [11:06:04] - memory: 16 GB
    
    
    [11:19:28] Model description:
    Final prediction for new objects (level 0) = 
    	 0.60805 * (5 averaged models Lvl_0_Pipe_0_Mod_0_LinearL2) +
    	 0.11170 * (5 averaged models Lvl_0_Pipe_1_Mod_0_LightGBM) +
    	 0.28024 * (5 averaged models Lvl_0_Pipe_1_Mod_1_Tuned_LightGBM) 
    
    Обучение завершено за 804.34 секунд
    ROC-AUC конфигурация 2: 0.875985
    Модель сохранена: src\lama_config2.pkl
    

### 5.3. Сравнение конфигураций LightAutoML

Сравниваем обе конфигурации по ROC AUC, потому что это метрика соревнования. Параллельно смотрим на время обучения, чтобы понимать, насколько дорого обходится улучшение. Из результатов видно что усложнив конфигурацию принципиальных различий в результатах нет, только значительно увеличилось время обучения, поэтому выберем простой вариант.


```python
results_comparison = pd.DataFrame({
    'Конфигурация': ['LightAutoML Config 1', 'LightAutoML Config 2'],
    'ROC-AUC': [score1, score2],
    'Время обучения (сек)': [train_time1, train_time2]
})

print(results_comparison)

best_automl = automl1 if score1 >= score2 else automl2
best_score = max(score1, score2)
best_config = 'Config 1' if score1 >= score2 else 'Config 2'

print(f"\nЛучшая конфигурация: {best_config}")
print(f"Лучший ROC-AUC: {best_score:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(['Config 1', 'Config 2'], [score1, score2], color=['skyblue', 'salmon'])
axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('Сравнение ROC-AUC')
axes[0].set_ylim([min(score1, score2) - 0.01, max(score1, score2) + 0.01])
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([score1, score2]):
    axes[0].text(i, v, f'{v:.6f}', ha='center', va='bottom')

axes[1].bar(['Config 1', 'Config 2'], [train_time1, train_time2], color=['lightgreen', 'orange'])
axes[1].set_ylabel('Время (секунды)')
axes[1].set_title('Время обучения')
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([train_time1, train_time2]):
    axes[1].text(i, v, f'{v:.1f}с', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(OTHER_DIR / 'baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

baseline_metrics = {
    'config1_auc': float(score1),
    'config2_auc': float(score2),
    'best_config': best_config,
    'best_auc': float(best_score)
}
import json
with open(PARAMS_DIR / 'baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f, indent=2)
```

               Конфигурация   ROC-AUC  Время обучения (сек)
    0  LightAutoML Config 1  0.875943            365.300634
    1  LightAutoML Config 2  0.875985            804.338054
    
    Лучшая конфигурация: Config 2
    Лучший ROC-AUC: 0.875985
    


    
![png](readme_files/readme_32_1.png)
    


### 5.4. Обучение лучшей модели на всех данных

После выбора лучшей конфигурации обучаем ее на всем train, потому что для финального решения важно использовать максимум данных. Для полного датасета увеличиваем лимит по времени, иначе обучение может не успеть нормально завершиться.


```python
print("Обучение лучшей модели LightAutoML на всех данных")

train_data_full = pd.read_csv(DATA_DIR / 'train.csv')
train_lam_full = train_data_full.drop(['id'], axis=1, errors='ignore')
train_lam_full = train_lam_full.reset_index(drop=True)

print(f"Размер полного датасета: {len(train_lam_full)} строк")

print("Обучение LightAutoML на полном датасете")
start_time = time.time()

# без cb, не работает на всем датасете(
automl_final = TabularAutoML(
    task=task,
    timeout=TIMEOUT*5,
    cpu_limit=N_THREADS,
    reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
    general_params={'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]}
)

oof_pred_final = automl_final.fit_predict(train_lam_full, roles=roles, verbose=1)

train_time_final = time.time() - start_time
print(f"Обучение завершено за {train_time_final:.2f} секунд")

joblib.dump(automl_final, MODELS_DIR / 'lama_final.pkl')
print(f"Финальная модель сохранена: {MODELS_DIR / 'lama_final.pkl'}")
```

    Обучение лучшей модели LightAutoML на всех данных
    Размер полного датасета: 11504798 строк
    Обучение LightAutoML на полном датасете
    [12:19:19] Stdout logging level is INFO.
    [12:19:19] Task: binary
    
    [12:19:19] Start automl preset with listed constraints:
    [12:19:19] - time: 9000.00 seconds
    [12:19:19] - CPU: 16 cores
    [12:19:19] - memory: 16 GB
    
    [13:04:21] Model description:
    Final prediction for new objects (level 0) = 
    	 0.52374 * (5 averaged models Lvl_0_Pipe_0_Mod_0_LinearL2) +
    	 0.47626 * (5 averaged models Lvl_0_Pipe_1_Mod_1_Tuned_LightGBM) 
    
    Обучение завершено за 2701.75 секунд
    Финальная модель сохранена: src\lama_final.pkl
    

### 5.5. Генерация submission для baseline

Делаем предсказания на test и сохраняем их в CSV. Используем финальную модель, обученную на всех данных, чтобы получить максимально возможное качество для отправки.
В результате на закрытых тестах бейзлайн модель показала хороший результат `0.87945`, лучше чем у примерно половины участников 


```python
test_data_full = pd.read_csv(DATA_DIR / 'test.csv')
print(f"Размер тестовой выборки: {len(test_data_full)}")

test_pred = automl_final.predict(test_data_full)
test_pred_proba = test_pred.data[:, 0] if test_pred.data.ndim > 1 else test_pred.data

submission_baseline = pd.DataFrame({
    'id': test_data_full['id'].values,
    'Response': test_pred_proba
})

submission_file_baseline = SUBMIT_DIR / 'baseline_submission.csv'
submission_baseline.to_csv(submission_file_baseline, index=False)

print(f"Submission создан: {submission_file_baseline}")
print(f"Размер: {submission_baseline.shape}")
print(f"Диапазон предсказаний: [{test_pred_proba.min():.6f}, {test_pred_proba.max():.6f}]")
print(f"Среднее предсказание: {test_pred_proba.mean():.6f}")
```

    Размер тестовой выборки: 7669866
    Submission создан: src\baseline_submission.csv
    Размер: (7669866, 2)
    Диапазон предсказаний: [0.000010, 0.929261]
    Среднее предсказание: 0.122994
    

## 6. Собственное решение без LightAutoML

В этом разделе показываем собственные решения без LightAutoML. Делаем несколько экспериментов от простых к сложным чтобы найти лучшее решение. Артефакты блокнота сохраняются в `src/`.

### 6.1. Pipeline версии 1: Базовый подход

Начинаем с простого пайплайна на RandomForest, добавляем пару простых признаков из EDA, логарифмируем Annual_Premium из-за сильной асимметрии, делаем группы по возрасту и добавляем взаимодействие возраста с Premium, потому что вместе эти признаки часто несут больше смысла, чем по отдельности. Категории кодируем числами, пропуски заполняем медианой, чтобы не ломать обучение. Параметры RandomForest берем умеренные и ограничиваем глубину, чтобы модель меньше переобучалась и работала стабильно.


```python
def create_features_v1(df):
    df = df.copy()
    
    # Логарифмирование
    if 'Annual_Premium' in df.columns:
        df['Annual_Premium_log'] = np.log1p(df['Annual_Premium'])
    
    # Категоризация
    if 'Age' in df.columns:
        df['Age_group'] = pd.cut(df['Age'], 
                                bins=[0, 25, 35, 45, 55, 100],
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        df['Age_group'] = df['Age_group'].astype(str)
    
    # Взаимодействие Age и Annual_Premium
    if 'Age' in df.columns and 'Annual_Premium' in df.columns:
        df['Age_Premium_interaction'] = df['Age'] * df['Annual_Premium'] / 1000
    
    return df

def preprocess_data_v1(X_train, X_val):
    X_train_proc = create_features_v1(X_train)
    X_val_proc = create_features_v1(X_val)
    
    categorical_cols = X_train_proc.select_dtypes(include=['object']).columns.tolist()
    
    # Label Encoding для категориальных
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_proc[col] = le.fit_transform(X_train_proc[col].astype(str))
        X_val_proc[col] = le.transform(X_val_proc[col].astype(str))
        label_encoders[col] = le
    
    # Заполнение пропусков
    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        median_val = X_train_proc[col].median()
        X_train_proc[col].fillna(median_val, inplace=True)
        X_val_proc[col].fillna(median_val, inplace=True)
    
    return X_train_proc, X_val_proc, label_encoders

print("Предобработка данных Pipeline v1")
X_train_proc_v1, X_val_proc_v1, encoders_v1 = preprocess_data_v1(X_train, X_val)

sample_size = 2000000
if len(X_train_proc_v1) > sample_size:
    sample_idx = np.random.choice(len(X_train_proc_v1), sample_size, replace=False)
    X_train_sample_v1 = X_train_proc_v1.iloc[sample_idx]
    y_train_sample_v1 = y_train.iloc[sample_idx]
    print(f"Используется выборка из {sample_size} строк")
else:
    X_train_sample_v1 = X_train_proc_v1
    y_train_sample_v1 = y_train

print("Обучение Pipeline v1")
start_time = time.time()

pipeline_v1 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=N_THREADS,
        verbose=0
    ))
])

pipeline_v1.fit(X_train_sample_v1, y_train_sample_v1)

train_time_v1 = time.time() - start_time
print(f"Обучение завершено за {train_time_v1:.2f} секунд")

val_pred_proba_v1 = pipeline_v1.predict_proba(X_val_proc_v1)[:, 1]
score_v1 = roc_auc_score(y_val, val_pred_proba_v1)

print(f"ROC-AUC Pipeline v1: {score_v1:.6f}")

joblib.dump(pipeline_v1, MODELS_DIR / 'pipeline_v1.pkl')
print(f"Модель сохранена: {MODELS_DIR / 'pipeline_v1.pkl'}")
```

    Предобработка данных Pipeline v1
    Используется выборка из 2000000 строк
    Обучение Pipeline v1
    Обучение завершено за 31.71 секунд
    ROC-AUC Pipeline v1: 0.860628
    Модель сохранена: src\models\pipeline_v1.pkl
    

Результат `0.86063` чуть хуже чем `0.87945` у LightAutoML

### 6.2. Pipeline версии 2: LightGBM подход

Во второй попытке переходим на бустинг, потому что для табличных данных он часто дает лучшее качество, чем RandomForest. Перед обучением делаем более аккуратную подготовку данных: убираем дубликаты, пробуем мягко отфильтровать явные аномалии и добавляем несколько дополнительных признаков, которые подсказал EDA. Дальше обучаем LightGBM и учитываем дисбаланс классов.


```python
print("Загрузка модели Pipeline v2")

import lightgbm as lgb

model_path = MODELS_DIR / 'model_v2_lgb.pkl'

def remove_duplicates(X, y=None):
    initial_size = len(X)
    if y is not None:
        X_with_target = X.copy()
        X_with_target['_target'] = y.values
        X_with_target = X_with_target.drop_duplicates()
        y_cleaned = pd.Series(X_with_target['_target'].values, index=X_with_target.index)
        X_cleaned = X_with_target.drop('_target', axis=1)
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned, y_cleaned
    else:
        X_cleaned = X.drop_duplicates()
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned

def detect_and_remove_anomalies(X, y=None, z_threshold=4.0, iqr_factor=2.5):
    initial_size = len(X)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_indices = set()
    
    for col in numeric_cols:
        col_data = X[col].dropna()
        if len(col_data) == 0:
            continue
        z_scores = np.abs(stats.zscore(col_data))
        z_outliers = col_data.index[z_scores > z_threshold].tolist()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        iqr_outliers = col_data.index[(col_data < lower_bound) | (col_data > upper_bound)].tolist()
        outliers = set(z_outliers) & set(iqr_outliers)
        anomaly_indices.update(outliers)
    
    if len(anomaly_indices) > 0:
        anomaly_pct = len(anomaly_indices) / initial_size * 100
        if anomaly_pct < 5.0:
            X_cleaned = X.drop(index=anomaly_indices)
            if y is not None:
                y_cleaned = y.drop(index=anomaly_indices)
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned, y_cleaned
            else:
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned
    
    if y is not None:
        return X, y
    else:
        return X

def create_features_improved(df):
    df = df.copy()
    if 'Annual_Premium' in df.columns:
        df['Annual_Premium_log'] = np.log1p(df['Annual_Premium'])
        df['Annual_Premium_sqrt'] = np.sqrt(df['Annual_Premium'])
    if 'Age' in df.columns:
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        df['Age_group'] = df['Age_group'].astype(str)
        df['Age_squared'] = df['Age'] ** 2
        df['Age_cubed'] = df['Age'] ** 3
    if 'Age' in df.columns and 'Annual_Premium' in df.columns:
        df['Age_Premium_interaction'] = df['Age'] * df['Annual_Premium'] / 1000
        df['Age_Premium_ratio'] = df['Annual_Premium'] / (df['Age'] + 1)
    if 'Vehicle_Age' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Vehicle_combination'] = df['Vehicle_Age'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Previously_Insured' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Insured_Damage'] = df['Previously_Insured'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Region_Code' in df.columns and 'Policy_Sales_Channel' in df.columns:
        df['Region_Channel'] = df['Region_Code'].astype(str) + '_' + df['Policy_Sales_Channel'].astype(str)
    return df

def preprocess_data_improved(X_train, X_val):
    X_train_proc = create_features_improved(X_train)
    X_val_proc = create_features_improved(X_val)
    
    categorical_cols = X_train_proc.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_proc[col] = le.fit_transform(X_train_proc[col].astype(str))
        X_val_proc[col] = X_val_proc[col].astype(str)
        known_classes = set(le.classes_)
        X_val_proc.loc[~X_val_proc[col].isin(known_classes), col] = le.classes_[0]
        X_val_proc[col] = le.transform(X_val_proc[col])
        label_encoders[col] = le
    
    for col in numeric_cols:
        median_val = X_train_proc[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_train_proc[col] = X_train_proc[col].fillna(median_val)
        X_val_proc[col] = X_val_proc[col].fillna(median_val)
    
    return X_train_proc, X_val_proc

print("Предобработка данных")
X_train_clean, y_train_clean = remove_duplicates(X_train.copy(), y_train.copy())
X_train_clean, y_train_clean = detect_and_remove_anomalies(X_train_clean, y_train_clean)

X_train_proc, X_val_proc = preprocess_data_improved(X_train_clean, X_val.copy())

if model_path.exists():
    print("Загрузка модели")
    model_improved = joblib.load(model_path)
    
    val_pred_proba_improved = model_improved.predict(X_val_proc, num_iteration=model_improved.best_iteration)
    score_improved = roc_auc_score(y_val, val_pred_proba_improved)
    
    print(f"ROC-AUC на валидации (Улучшенная модель): {score_improved:.6f}")
else:
    print("Обучение модели")
    
    sample_size = 2000000
    if len(X_train_proc) > sample_size:
        sample_idx = np.random.choice(len(X_train_proc), sample_size, replace=False)
        X_train_sample = X_train_proc.iloc[sample_idx]
        y_train_sample = y_train_clean.iloc[sample_idx]
        print(f"Используется выборка из {sample_size} строк для обучения")
    else:
        X_train_sample = X_train_proc
        y_train_sample = y_train_clean
    
    train_data = lgb.Dataset(X_train_sample, label=y_train_sample)
    val_data = lgb.Dataset(X_val_proc, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'scale_pos_weight': (y_train_sample == 0).sum() / (y_train_sample == 1).sum()
    }
    
    model_improved = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_pred_proba_improved = model_improved.predict(X_val_proc, num_iteration=model_improved.best_iteration)
    score_improved = roc_auc_score(y_val, val_pred_proba_improved)
    print(f"ROC-AUC на валидации (Улучшенная модель): {score_improved:.6f}")
    
    joblib.dump(model_improved, MODELS_DIR / 'model_improved_optuna.pkl')

print("Сравнение результатов")
print(f"Pipeline v1: {score_v1:.6f}")
if score_improved is not None:
    print(f"Улучшенный подход: {score_improved:.6f}")
```

    Загрузка модели Pipeline v2
    Предобработка данных
      Удалено аномалий: 20,574 (0.22%)
    Обучение модели
    Используется выборка из 2000000 строк для обучения
    ROC-AUC на валидации (Улучшенная модель): 0.877647
    Сравнение результатов
    Pipeline v1: 0.860565
    Улучшенный подход: 0.877647
    

Результат `0.87765` близок к `0.87945` у LightAutoML, но при этом на порядок быстрее

### 6.3.1 Улучшенный LightGBM с очисткой данных и Optuna

Здесь собираем воедино все идеи из предыдущих шагов и добавляем автоматический подбор гиперпараметров через Optuna. Сначала очищаем train от дубликатов и явных аномалий, затем делаем feature engineering, после чего на стратифицированной подвыборке запускаем Optuna, чтобы найти хорошие параметры. Лучшие настройки переносим на полный очищенный train+val и обучаем финальную модель


```python
print("Улучшенный LightGBM с Optuna оптимизацией")

import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split

model_path = MODELS_DIR / 'model_improved_optuna.pkl'

def remove_duplicates_optuna(X, y=None):
    initial_size = len(X)
    if y is not None:
        X_with_target = X.copy()
        X_with_target['_target'] = y.values
        X_with_target = X_with_target.drop_duplicates()
        y_cleaned = pd.Series(X_with_target['_target'].values, index=X_with_target.index)
        X_cleaned = X_with_target.drop('_target', axis=1)
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned, y_cleaned
    else:
        X_cleaned = X.drop_duplicates()
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned

def detect_and_remove_anomalies_optuna(X, y=None, z_threshold=4.0, iqr_factor=2.5):
    initial_size = len(X)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_indices = set()
    
    for col in numeric_cols:
        col_data = X[col].dropna()
        if len(col_data) == 0:
            continue
        z_scores = np.abs(stats.zscore(col_data))
        z_outliers = col_data.index[z_scores > z_threshold].tolist()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        iqr_outliers = col_data.index[(col_data < lower_bound) | (col_data > upper_bound)].tolist()
        outliers = set(z_outliers) & set(iqr_outliers)
        anomaly_indices.update(outliers)
    
    if len(anomaly_indices) > 0:
        anomaly_pct = len(anomaly_indices) / initial_size * 100
        if anomaly_pct < 5.0:
            X_cleaned = X.drop(index=anomaly_indices)
            if y is not None:
                y_cleaned = y.drop(index=anomaly_indices)
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned, y_cleaned
            else:
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned
    
    if y is not None:
        return X, y
    else:
        return X

def create_features_optuna(df):
    df = df.copy()
    if 'Annual_Premium' in df.columns:
        df['Annual_Premium_log'] = np.log1p(df['Annual_Premium'])
        df['Annual_Premium_sqrt'] = np.sqrt(df['Annual_Premium'])
    if 'Age' in df.columns:
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        df['Age_group'] = df['Age_group'].astype(str)
        df['Age_squared'] = df['Age'] ** 2
        df['Age_cubed'] = df['Age'] ** 3
    if 'Age' in df.columns and 'Annual_Premium' in df.columns:
        df['Age_Premium_interaction'] = df['Age'] * df['Annual_Premium'] / 1000
        df['Age_Premium_ratio'] = df['Annual_Premium'] / (df['Age'] + 1)
    if 'Vehicle_Age' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Vehicle_combination'] = df['Vehicle_Age'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Previously_Insured' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Insured_Damage'] = df['Previously_Insured'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Region_Code' in df.columns and 'Policy_Sales_Channel' in df.columns:
        df['Region_Channel'] = df['Region_Code'].astype(str) + '_' + df['Policy_Sales_Channel'].astype(str)
    return df

def preprocess_data_optuna(X_train, X_val):
    X_train_proc = create_features_optuna(X_train)
    X_val_proc = create_features_optuna(X_val)
    
    categorical_cols = X_train_proc.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_proc[col] = le.fit_transform(X_train_proc[col].astype(str))
        X_val_proc[col] = X_val_proc[col].astype(str)
        known_classes = set(le.classes_)
        X_val_proc.loc[~X_val_proc[col].isin(known_classes), col] = le.classes_[0]
        X_val_proc[col] = le.transform(X_val_proc[col])
        label_encoders[col] = le
    
    for col in numeric_cols:
        median_val = X_train_proc[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_train_proc[col] = X_train_proc[col].fillna(median_val)
        X_val_proc[col] = X_val_proc[col].fillna(median_val)
    
    return X_train_proc, X_val_proc, label_encoders

def coreset_sampling_stratified(X, y, n_samples=300000):
    if len(X) <= n_samples:
        print(f"Размер данных ({len(X):,}) <= требуемого размера ({n_samples:,}), выборка не требуется")
        return X, y
    
    _, X_sample, _, y_sample = train_test_split(
        X, y,
        train_size=n_samples,
        stratify=y,
        random_state=42
    )
    print(f"Стратифицированная выборка: {len(X_sample):,} строк")
    return X_sample, y_sample

print("\n[1/4] Очистка данных")
X_train_clean, y_train_clean = remove_duplicates_optuna(X_train.copy(), y_train.copy())
X_train_clean, y_train_clean = detect_and_remove_anomalies_optuna(X_train_clean, y_train_clean)

print("\n[2/4] Предобработка данных")
X_train_proc, X_val_proc, label_encoders = preprocess_data_optuna(X_train_clean, X_val.copy())

print("\n[3/4] Стратифицированная выборка для Optuna")
optuna_sample_size = 300000
X_train_optuna, y_train_optuna = coreset_sampling_stratified(X_train_proc, y_train_clean, n_samples=optuna_sample_size)

print("\n[4/4] Оптимизация гиперпараметров с Optuna")

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'verbose': -1,
        'random_state': 42,
        'n_jobs': N_THREADS,
        'scale_pos_weight': (y_train_optuna == 0).sum() / (y_train_optuna == 1).sum()
    }
    
    train_data = lgb.Dataset(X_train_optuna, label=y_train_optuna)
    val_data = lgb.Dataset(X_val_proc, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    y_pred = model.predict(X_val_proc, num_iteration=model.best_iteration)
    score = roc_auc_score(y_val, y_pred)
    
    return score

study = optuna.create_study(
    direction='maximize',
    study_name='lgb_optimization_improved',
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print("результат оптимизации:")
print(f"Лучший ROC-AUC: {study.best_value:.6f}")
print(f"Лучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"{'='*60}\n")

print("\nОбучение финальной модели на полном очищенном датасете")

best_params = study.best_params.copy()
best_params['scale_pos_weight'] = (y_train_clean == 0).sum() / (y_train_clean == 1).sum()
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['boosting_type'] = 'gbdt'
best_params['verbose'] = -1
best_params['random_state'] = 42
best_params['n_jobs'] = N_THREADS

train_data_final = lgb.Dataset(X_train_proc, label=y_train_clean)
val_data_final = lgb.Dataset(X_val_proc, label=y_val, reference=train_data_final)

model_final = lgb.train(
    best_params,
    train_data_final,
    valid_sets=[val_data_final],
    num_boost_round=3000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=True),
        lgb.log_evaluation(period=200)
    ]
)

val_pred_proba = model_final.predict(X_val_proc, num_iteration=model_final.best_iteration)
score_optuna = roc_auc_score(y_val, val_pred_proba)

print(f"ROC-AUC на валидации: {score_optuna:.6f}")

joblib.dump(model_final, model_path)
print(f"Модель сохранена: {model_path}")

print("\nСравнение результатов:")
if 'score_v1' in globals():
    print(f"Pipeline v1 (RandomForest): {score_v1:.6f}")
if 'score_improved' in globals():
    print(f"Улучшенный подход (6.2): {score_improved:.6f}")
print(f"Улучшенный подход с Optuna (6.3): {score_optuna:.6f}")
```

    Улучшенный LightGBM с Optuna оптимизацией
    
    [1/4] Очистка данных
    Удалено аномалий: 20,574 (0.22%)
    
    [2/4] Предобработка данных
    
    [3/4] Стратифицированная выборка для Optuna
    Стратифицированная выборка: 8,883,264 строк
    
    [4/4] Оптимизация гиперпараметров с Optuna

    Best trial: 44. Best value: 0.880691: 100%|██████████| 50/50 [2:05:55<00:00, 151.11s/it]
    

    [I 2025-12-18 16:52:14,316] Trial 49 finished with value: 0.8806206385296312 and parameters: {'num_leaves': 122, 'learning_rate': 0.1300651327277901, 'feature_fraction': 0.6335654026761135, 'bagging_fraction': 0.8270675787847714, 'bagging_freq': 10, 'min_child_samples': 164, 'reg_alpha': 5.077338711929052e-05, 'reg_lambda': 2.800909454542818e-05, 'max_depth': 8, 'min_split_gain': 0.8756661413740314}. Best is trial 44 with value: 0.8806908922507936.
    результат оптимизации:
    Лучший ROC-AUC: 0.880691
    Лучшие параметры:
      num_leaves: 96
      learning_rate: 0.10162578711558443
      feature_fraction: 0.7798149581817775
      bagging_fraction: 0.8366936126528723
      bagging_freq: 8
      min_child_samples: 200
      reg_alpha: 9.352896985399261e-07
      reg_lambda: 0.04355479172757652
      max_depth: 8
      min_split_gain: 0.8016739053601484
    ============================================================
    
    
    Обучение финальной модели на полном очищенном датасете
    Training until validation scores don't improve for 150 rounds
    [200]	valid_0's auc: 0.879108
    [400]	valid_0's auc: 0.880043
    [600]	valid_0's auc: 0.880336
    [800]	valid_0's auc: 0.880475
    [1000]	valid_0's auc: 0.88052
    [1200]	valid_0's auc: 0.880548
    Early stopping, best iteration is:
    [1184]	valid_0's auc: 0.880549
    ROC-AUC на валидации: 0.880549
    Модель сохранена: src\model_improved_optuna.pkl
    
    Сравнение результатов:
    Pipeline v1 (RandomForest): 0.860565
    Улучшенный подход (6.2): 0.877647
    Улучшенный подход с Optuna (6.3): 0.880549
    

Отличный Результат `0.88055` даже чуть лучше чем `0.87945` у LightAutoML, но при этом заняло вдвое больше времени

### 6.3.2 Улучшенный LightGBM с очисткой данных и Optuna + оптимизация по num_boost_round (500-4000)

```python
print("Улучшенный LightGBM с Optuna оптимизацией + num_boost_round")

import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split

model_path = MODELS_DIR / 'model_improved_optuna.pkl'

def remove_duplicates_optuna(X, y=None):
    initial_size = len(X)
    if y is not None:
        X_with_target = X.copy()
        X_with_target['_target'] = y.values
        X_with_target = X_with_target.drop_duplicates()
        y_cleaned = pd.Series(X_with_target['_target'].values, index=X_with_target.index)
        X_cleaned = X_with_target.drop('_target', axis=1)
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned, y_cleaned
    else:
        X_cleaned = X.drop_duplicates()
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned

def detect_and_remove_anomalies_optuna(X, y=None, z_threshold=4.0, iqr_factor=2.5):
    initial_size = len(X)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_indices = set()
    
    for col in numeric_cols:
        col_data = X[col].dropna()
        if len(col_data) == 0:
            continue
        z_scores = np.abs(stats.zscore(col_data))
        z_outliers = col_data.index[z_scores > z_threshold].tolist()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        iqr_outliers = col_data.index[(col_data < lower_bound) | (col_data > upper_bound)].tolist()
        outliers = set(z_outliers) & set(iqr_outliers)
        anomaly_indices.update(outliers)
    
    if len(anomaly_indices) > 0:
        anomaly_pct = len(anomaly_indices) / initial_size * 100
        if anomaly_pct < 5.0:
            X_cleaned = X.drop(index=anomaly_indices)
            if y is not None:
                y_cleaned = y.drop(index=anomaly_indices)
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned, y_cleaned
            else:
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned
    
    if y is not None:
        return X, y
    else:
        return X

def create_features_optuna(df):
    df = df.copy()
    if 'Annual_Premium' in df.columns:
        df['Annual_Premium_log'] = np.log1p(df['Annual_Premium'])
        df['Annual_Premium_sqrt'] = np.sqrt(df['Annual_Premium'])
    if 'Age' in df.columns:
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        df['Age_group'] = df['Age_group'].astype(str)
        df['Age_squared'] = df['Age'] ** 2
        df['Age_cubed'] = df['Age'] ** 3
    if 'Age' in df.columns and 'Annual_Premium' in df.columns:
        df['Age_Premium_interaction'] = df['Age'] * df['Annual_Premium'] / 1000
        df['Age_Premium_ratio'] = df['Annual_Premium'] / (df['Age'] + 1)
    if 'Vehicle_Age' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Vehicle_combination'] = df['Vehicle_Age'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Previously_Insured' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Insured_Damage'] = df['Previously_Insured'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Region_Code' in df.columns and 'Policy_Sales_Channel' in df.columns:
        df['Region_Channel'] = df['Region_Code'].astype(str) + '_' + df['Policy_Sales_Channel'].astype(str)
    return df

def preprocess_data_optuna(X_train, X_val):
    X_train_proc = create_features_optuna(X_train)
    X_val_proc = create_features_optuna(X_val)
    
    categorical_cols = X_train_proc.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_proc[col] = le.fit_transform(X_train_proc[col].astype(str))
        X_val_proc[col] = X_val_proc[col].astype(str)
        known_classes = set(le.classes_)
        X_val_proc.loc[~X_val_proc[col].isin(known_classes), col] = le.classes_[0]
        X_val_proc[col] = le.transform(X_val_proc[col])
        label_encoders[col] = le
    
    for col in numeric_cols:
        median_val = X_train_proc[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_train_proc[col] = X_train_proc[col].fillna(median_val)
        X_val_proc[col] = X_val_proc[col].fillna(median_val)
    
    return X_train_proc, X_val_proc, label_encoders

def coreset_sampling_stratified(X, y, n_samples=300000):
    _, X_sample, _, y_sample = train_test_split(
        X, y,
        train_size=n_samples,
        stratify=y,
        random_state=42
    )
    print(f"Стратифицированная выборка: {len(X_sample):,} строк")
    return X_sample, y_sample

print("\n[1/4] Очистка данных")
X_train_clean, y_train_clean = remove_duplicates_optuna(X_train.copy(), y_train.copy())
X_train_clean, y_train_clean = detect_and_remove_anomalies_optuna(X_train_clean, y_train_clean)

print("\n[2/4] Предобработка данных")
X_train_proc, X_val_proc, label_encoders = preprocess_data_optuna(X_train_clean, X_val.copy())

print("\n[3/4] Стратифицированная выборка для Optuna")
optuna_sample_size = 300000
X_train_optuna, y_train_optuna = coreset_sampling_stratified(X_train_proc, y_train_clean, n_samples=optuna_sample_size)

print("\n[4/4] Оптимизация гиперпараметров с Optuna")

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'verbose': -1,
        'random_state': 42,
        'n_jobs': N_THREADS,
        'scale_pos_weight': (y_train_optuna == 0).sum() / (y_train_optuna == 1).sum()
    }
    num_boost_round = trial.suggest_int('num_boost_round', 500, 4000)

    train_data = lgb.Dataset(X_train_optuna, label=y_train_optuna)
    val_data = lgb.Dataset(X_val_proc, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    y_pred = model.predict(X_val_proc, num_iteration=model.best_iteration)
    score = roc_auc_score(y_val, y_pred)
    
    return score

study = optuna.create_study(
    direction='maximize',
    study_name='lgb_optimization_improved',
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

print("результат оптимизации:")
print(f"Лучший ROC-AUC: {study.best_value:.6f}")
print(f"Лучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"{'='*60}\n")

print("\nОбучение финальной модели на полном очищенном датасете")

best_params = study.best_params.copy()
num_boost_round_final = best_params.pop('num_boost_round', 3000)

best_params['scale_pos_weight'] = (y_train_clean == 0).sum() / (y_train_clean == 1).sum()
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['boosting_type'] = 'gbdt'
best_params['verbose'] = -1
best_params['random_state'] = 42
best_params['n_jobs'] = N_THREADS

train_data_final = lgb.Dataset(X_train_proc, label=y_train_clean)
val_data_final = lgb.Dataset(X_val_proc, label=y_val, reference=train_data_final)

model_final = lgb.train(
    best_params,
    train_data_final,
    valid_sets=[val_data_final],
    num_boost_round=num_boost_round_final,
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=True),
        lgb.log_evaluation(period=200)
    ]
)

val_pred_proba = model_final.predict(X_val_proc, num_iteration=model_final.best_iteration)
score_optuna = roc_auc_score(y_val, val_pred_proba)

print(f"ROC-AUC на валидации: {score_optuna:.6f}")

joblib.dump(model_final, model_path)
print(f"Модель сохранена: {model_path}")
```
Улучшенный LightGBM с Optuna оптимизацией + num_boost_round

[1/4] Очистка данных
Удалено аномалий: 20,574 (0.22%)

[2/4] Предобработка данных

[3/4] Стратифицированная выборка для Optuna
Стратифицированная выборка: 8,883,264 строк

[4/4] Оптимизация гиперпараметров с Optuna
Best trial: 0. Best value: 0.879779:   2%|▏         | 1/50 [01:01<50:06, 61.36s/it]
[I 2025-12-29 10:34:24,147] Trial 0 finished with value: 0.8797786016991349 and parameters: {'num_leaves': 69, 'learning_rate': 0.24517932047070642, 'feature_fraction': 0.839196365086843, 'bagging_fraction': 0.759195090518222, 'bagging_freq': 2, 'min_child_samples': 35, 'reg_alpha': 3.3323645788192616e-08, 'reg_lambda': 0.6245760287469893, 'max_depth': 14, 'min_split_gain': 0.7080725777960455, 'num_boost_round': 572}. Best is trial 0 with value: 0.8797786016991349.
Best trial: 1. Best value: 0.879855:   4%|▍         | 2/50 [02:15<55:16, 69.10s/it]
[I 2025-12-29 10:35:38,663] Trial 1 finished with value: 0.8798554648159742 and parameters: {'num_leaves': 147, 'learning_rate': 0.15107024270948044, 'feature_fraction': 0.5274034664069657, 'bagging_fraction': 0.5090949803242604, 'bagging_freq': 2, 'min_child_samples': 64, 'reg_alpha': 0.00052821153945323, 'reg_lambda': 7.71800699380605e-05, 'max_depth': 9, 'min_split_gain': 0.6118528947223795, 'num_boost_round': 988}. Best is trial 1 with value: 0.8798554648159742.
Best trial: 1. Best value: 0.879855:   6%|▌         | 3/50 [05:03<1:29:12, 113.88s/it]
[I 2025-12-29 10:38:25,834] Trial 2 finished with value: 0.8787048263700219 and parameters: {'num_leaves': 58, 'learning_rate': 0.02240870575939826, 'feature_fraction': 0.6736419905302216, 'bagging_fraction': 0.8711055768358081, 'bagging_freq': 2, 'min_child_samples': 105, 'reg_alpha': 0.0021465011216654484, 'reg_lambda': 2.6185068507773707e-08, 'max_depth': 14, 'min_split_gain': 0.17052412368729153, 'num_boost_round': 727}. Best is trial 1 with value: 0.8798554648159742.
Best trial: 1. Best value: 0.879855:   8%|▊         | 4/50 [05:23<59:04, 77.06s/it]   
[I 2025-12-29 10:38:46,450] Trial 3 finished with value: 0.8777922109429241 and parameters: {'num_leaves': 144, 'learning_rate': 0.260621242754743, 'feature_fraction': 0.8850384088698766, 'bagging_fraction': 0.5827682615040224, 'bagging_freq': 1, 'min_child_samples': 139, 'reg_alpha': 9.148975058772307e-05, 'reg_lambda': 1.254134495897175e-07, 'max_depth': 12, 'min_split_gain': 0.034388521115218396, 'num_boost_round': 3683}. Best is trial 1 with value: 0.8798554648159742.
Best trial: 4. Best value: 0.880624:  10%|█         | 5/50 [11:42<2:19:20, 185.78s/it]
[I 2025-12-29 10:45:05,008] Trial 4 finished with value: 0.8806236958931424 and parameters: {'num_leaves': 53, 'learning_rate': 0.07534159891754702, 'feature_fraction': 0.5870266456536466, 'bagging_fraction': 0.7120408127066865, 'bagging_freq': 6, 'min_child_samples': 41, 'reg_alpha': 5.324289357128436, 'reg_lambda': 0.09466630153726856, 'max_depth': 20, 'min_split_gain': 0.8948273504276488, 'num_boost_round': 2593}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  12%|█▏        | 6/50 [19:07<3:21:00, 274.10s/it]
[I 2025-12-29 10:52:30,541] Trial 5 finished with value: 0.8797300578221839 and parameters: {'num_leaves': 140, 'learning_rate': 0.007183284336890004, 'feature_fraction': 0.5175897174514872, 'bagging_fraction': 0.4271363733463229, 'bagging_freq': 4, 'min_child_samples': 81, 'reg_alpha': 2.7678419414850017e-06, 'reg_lambda': 0.28749982347407854, 'max_depth': 10, 'min_split_gain': 0.28093450968738076, 'num_boost_round': 2399}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  14%|█▍        | 7/50 [22:39<3:01:46, 253.64s/it]
[I 2025-12-29 10:56:02,058] Trial 6 finished with value: 0.8804372693292772 and parameters: {'num_leaves': 38, 'learning_rate': 0.13347427443576154, 'feature_fraction': 0.44473038620786254, 'bagging_fraction': 0.9921321619603104, 'bagging_freq': 8, 'min_child_samples': 43, 'reg_alpha': 1.1212412169964432e-08, 'reg_lambda': 0.2183498289760726, 'max_depth': 16, 'min_split_gain': 0.7290071680409873, 'num_boost_round': 3200}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  16%|█▌        | 8/50 [31:17<3:56:37, 338.03s/it]
[I 2025-12-29 11:04:40,771] Trial 7 finished with value: 0.8796230464397168 and parameters: {'num_leaves': 29, 'learning_rate': 0.02169583092556068, 'feature_fraction': 0.4695214357150779, 'bagging_fraction': 0.9178620555253562, 'bagging_freq': 7, 'min_child_samples': 69, 'reg_alpha': 3.732717755563729e-08, 'reg_lambda': 6.292756043818863e-06, 'max_depth': 10, 'min_split_gain': 0.7296061783380641, 'num_boost_round': 2732}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  18%|█▊        | 9/50 [33:28<3:06:38, 273.13s/it]
[I 2025-12-29 11:06:51,204] Trial 8 finished with value: 0.8800017829064629 and parameters: {'num_leaves': 136, 'learning_rate': 0.034565238985787616, 'feature_fraction': 0.471756547562981, 'bagging_fraction': 0.827946872333797, 'bagging_freq': 8, 'min_child_samples': 115, 'reg_alpha': 0.08683696167603723, 'reg_lambda': 0.0002780739892288472, 'max_depth': 13, 'min_split_gain': 0.42754101835854963, 'num_boost_round': 588}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  20%|██        | 10/50 [35:31<2:31:09, 226.74s/it]
[I 2025-12-29 11:08:54,063] Trial 9 finished with value: 0.8711297214797055 and parameters: {'num_leaves': 34, 'learning_rate': 0.0056866415006646374, 'feature_fraction': 0.7818462467582683, 'bagging_fraction': 0.588613588645796, 'bagging_freq': 6, 'min_child_samples': 182, 'reg_alpha': 1.7523871598466864e-06, 'reg_lambda': 4.9368087974032924e-05, 'max_depth': 17, 'min_split_gain': 0.22879816549162246, 'num_boost_round': 769}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  22%|██▏       | 11/50 [38:21<2:16:04, 209.35s/it]
[I 2025-12-29 11:11:43,980] Trial 10 finished with value: 0.8802120080550238 and parameters: {'num_leaves': 114, 'learning_rate': 0.08496662763968879, 'feature_fraction': 0.9817222664727194, 'bagging_fraction': 0.7133770405755714, 'bagging_freq': 10, 'min_child_samples': 12, 'reg_alpha': 5.724031268879675, 'reg_lambda': 0.012278672550406745, 'max_depth': 20, 'min_split_gain': 0.9923626654603367, 'num_boost_round': 1482}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  24%|██▍       | 12/50 [41:45<2:11:42, 207.96s/it]
[I 2025-12-29 11:15:08,762] Trial 11 finished with value: 0.8804494896982779 and parameters: {'num_leaves': 52, 'learning_rate': 0.09115080816126353, 'feature_fraction': 0.6364067449384899, 'bagging_fraction': 0.9915758251273215, 'bagging_freq': 9, 'min_child_samples': 7, 'reg_alpha': 7.523537286414183, 'reg_lambda': 0.008420423778075163, 'max_depth': 20, 'min_split_gain': 0.9345044878402002, 'num_boost_round': 3268}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  26%|██▌       | 13/50 [44:53<2:04:26, 201.81s/it]
[I 2025-12-29 11:18:16,423] Trial 12 finished with value: 0.8805781265339483 and parameters: {'num_leaves': 93, 'learning_rate': 0.07236622141212541, 'feature_fraction': 0.6486310158651231, 'bagging_fraction': 0.9944431202010833, 'bagging_freq': 10, 'min_child_samples': 6, 'reg_alpha': 8.236611404863314, 'reg_lambda': 0.008865465164105238, 'max_depth': 20, 'min_split_gain': 0.9827429795601963, 'num_boost_round': 3157}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  28%|██▊       | 14/50 [50:09<2:21:46, 236.28s/it]
[I 2025-12-29 11:23:32,354] Trial 13 finished with value: 0.8799139753399625 and parameters: {'num_leaves': 94, 'learning_rate': 0.05187206227880954, 'feature_fraction': 0.6096840026527188, 'bagging_fraction': 0.6100503603842299, 'bagging_freq': 5, 'min_child_samples': 32, 'reg_alpha': 0.10844047920954882, 'reg_lambda': 9.52546993498323, 'max_depth': 5, 'min_split_gain': 0.8819520692001515, 'num_boost_round': 1855}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  30%|███       | 15/50 [50:58<1:44:53, 179.80s/it]
[I 2025-12-29 11:24:21,266] Trial 14 finished with value: 0.8784018793916037 and parameters: {'num_leaves': 91, 'learning_rate': 0.059641177332802435, 'feature_fraction': 0.7404681632046692, 'bagging_fraction': 0.8074571382520241, 'bagging_freq': 10, 'min_child_samples': 6, 'reg_alpha': 0.39479425332602563, 'reg_lambda': 0.006919225887522044, 'max_depth': 18, 'min_split_gain': 0.8305216333158655, 'num_boost_round': 2756}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  32%|███▏      | 16/50 [1:00:36<2:49:48, 299.67s/it]
[I 2025-12-29 11:33:59,299] Trial 15 finished with value: 0.880570215717424 and parameters: {'num_leaves': 77, 'learning_rate': 0.03283485242690805, 'feature_fraction': 0.5889158463852501, 'bagging_fraction': 0.6760077383148502, 'bagging_freq': 4, 'min_child_samples': 48, 'reg_alpha': 0.01399482441401767, 'reg_lambda': 0.002197354906255265, 'max_depth': 18, 'min_split_gain': 0.5006901329435471, 'num_boost_round': 3809}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  34%|███▍      | 17/50 [1:07:48<3:06:43, 339.50s/it]
[I 2025-12-29 11:41:11,418] Trial 16 finished with value: 0.8804213124361573 and parameters: {'num_leaves': 110, 'learning_rate': 0.020607555725193776, 'feature_fraction': 0.7240570120681563, 'bagging_fraction': 0.9305768134610474, 'bagging_freq': 6, 'min_child_samples': 136, 'reg_alpha': 0.8233323501240152, 'reg_lambda': 4.865109964279932, 'max_depth': 20, 'min_split_gain': 0.876976738495405, 'num_boost_round': 2062}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  36%|███▌      | 18/50 [1:09:59<2:27:40, 276.88s/it]
[I 2025-12-29 11:43:22,516] Trial 17 finished with value: 0.8800879688019332 and parameters: {'num_leaves': 105, 'learning_rate': 0.1414598449222712, 'feature_fraction': 0.5602662398885422, 'bagging_fraction': 0.6642628205405887, 'bagging_freq': 8, 'min_child_samples': 89, 'reg_alpha': 0.011484718160115693, 'reg_lambda': 0.07022576529008312, 'max_depth': 16, 'min_split_gain': 0.9973065566821956, 'num_boost_round': 3131}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  38%|███▊      | 19/50 [1:14:24<2:21:14, 273.39s/it]
[I 2025-12-29 11:47:47,775] Trial 18 finished with value: 0.8803654204244029 and parameters: {'num_leaves': 52, 'learning_rate': 0.07020978476248484, 'feature_fraction': 0.6664840116054781, 'bagging_fraction': 0.40385404541155373, 'bagging_freq': 4, 'min_child_samples': 26, 'reg_alpha': 9.797958350741983, 'reg_lambda': 1.4477928778418085e-06, 'max_depth': 18, 'min_split_gain': 0.6040545949060365, 'num_boost_round': 2569}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  40%|████      | 20/50 [1:19:22<2:20:15, 280.51s/it]
[I 2025-12-29 11:52:44,892] Trial 19 finished with value: 0.8763379768970683 and parameters: {'num_leaves': 20, 'learning_rate': 0.013015760283633436, 'feature_fraction': 0.41722033803557923, 'bagging_fraction': 0.754419020461397, 'bagging_freq': 9, 'min_child_samples': 57, 'reg_alpha': 0.4320292735392467, 'reg_lambda': 0.0013128021407523044, 'max_depth': 6, 'min_split_gain': 0.7907319355022951, 'num_boost_round': 1688}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  42%|████▏     | 21/50 [1:26:10<2:34:06, 318.85s/it]
[I 2025-12-29 11:59:33,127] Trial 20 finished with value: 0.8803559424409797 and parameters: {'num_leaves': 68, 'learning_rate': 0.04600463962490479, 'feature_fraction': 0.834268428634075, 'bagging_fraction': 0.5328908338297378, 'bagging_freq': 5, 'min_child_samples': 200, 'reg_alpha': 1.938418664214207e-05, 'reg_lambda': 0.041718617992926005, 'max_depth': 15, 'min_split_gain': 0.4080086291387736, 'num_boost_round': 3998}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  44%|████▍     | 22/50 [1:36:13<3:08:35, 404.12s/it]
[I 2025-12-29 12:09:36,099] Trial 21 finished with value: 0.880563377667921 and parameters: {'num_leaves': 77, 'learning_rate': 0.03215096299893353, 'feature_fraction': 0.6021906000829224, 'bagging_fraction': 0.6767060534617074, 'bagging_freq': 4, 'min_child_samples': 50, 'reg_alpha': 0.01266421422431315, 'reg_lambda': 0.0012577658934891653, 'max_depth': 18, 'min_split_gain': 0.5660980335096722, 'num_boost_round': 3504}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  46%|████▌     | 23/50 [1:36:46<2:11:47, 292.86s/it]
[I 2025-12-29 12:10:09,447] Trial 22 finished with value: 0.8784382720472655 and parameters: {'num_leaves': 83, 'learning_rate': 0.10004176304484887, 'feature_fraction': 0.5707647879433421, 'bagging_fraction': 0.7651635032186925, 'bagging_freq': 3, 'min_child_samples': 24, 'reg_alpha': 0.9650035038231072, 'reg_lambda': 0.0013119016709877238, 'max_depth': 19, 'min_split_gain': 0.47806217945032947, 'num_boost_round': 3901}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 4. Best value: 0.880624:  48%|████▊     | 24/50 [1:45:17<2:35:16, 358.32s/it]
[I 2025-12-29 12:18:40,486] Trial 23 finished with value: 0.8805428853131856 and parameters: {'num_leaves': 100, 'learning_rate': 0.03137191338256602, 'feature_fraction': 0.6453993527919241, 'bagging_fraction': 0.6325103186466039, 'bagging_freq': 7, 'min_child_samples': 46, 'reg_alpha': 0.01788611744605737, 'reg_lambda': 0.710235800134321, 'max_depth': 19, 'min_split_gain': 0.35776043590341927, 'num_boost_round': 2968}. Best is trial 4 with value: 0.8806236958931424.
Best trial: 24. Best value: 0.880642:  50%|█████     | 25/50 [1:51:29<2:30:59, 362.39s/it]
[I 2025-12-29 12:24:52,366] Trial 24 finished with value: 0.8806421596068326 and parameters: {'num_leaves': 124, 'learning_rate': 0.04524618004931291, 'feature_fraction': 0.5200565453283603, 'bagging_fraction': 0.49332214738319846, 'bagging_freq': 3, 'min_child_samples': 77, 'reg_alpha': 2.037228958120651, 'reg_lambda': 0.0496808588081603, 'max_depth': 17, 'min_split_gain': 0.014540324017466688, 'num_boost_round': 3541}. Best is trial 24 with value: 0.8806421596068326.
Best trial: 24. Best value: 0.880642:  52%|█████▏    | 26/50 [1:54:55<2:06:08, 315.35s/it]
[I 2025-12-29 12:28:17,963] Trial 25 finished with value: 0.8804619177590547 and parameters: {'num_leaves': 127, 'learning_rate': 0.06637751041709844, 'feature_fraction': 0.517086833251891, 'bagging_fraction': 0.46904700298656005, 'bagging_freq': 1, 'min_child_samples': 79, 'reg_alpha': 1.7437350362442465, 'reg_lambda': 0.05021918965479869, 'max_depth': 16, 'min_split_gain': 0.09532926596955016, 'num_boost_round': 3353}. Best is trial 24 with value: 0.8806421596068326.
Best trial: 24. Best value: 0.880642:  54%|█████▍    | 27/50 [1:56:00<1:32:06, 240.28s/it]
[I 2025-12-29 12:29:23,082] Trial 26 finished with value: 0.8798468714506643 and parameters: {'num_leaves': 119, 'learning_rate': 0.18802802446191522, 'feature_fraction': 0.5425538251876587, 'bagging_fraction': 0.5575558433329104, 'bagging_freq': 3, 'min_child_samples': 23, 'reg_alpha': 0.10731727958113504, 'reg_lambda': 1.2853306293322286, 'max_depth': 19, 'min_split_gain': 0.8980490774240812, 'num_boost_round': 2944}. Best is trial 24 with value: 0.8806421596068326.
Best trial: 24. Best value: 0.880642:  56%|█████▌    | 28/50 [2:01:04<1:35:07, 259.43s/it]
[I 2025-12-29 12:34:27,217] Trial 27 finished with value: 0.8803541423557044 and parameters: {'num_leaves': 126, 'learning_rate': 0.04736867465706314, 'feature_fraction': 0.7147003108075001, 'bagging_fraction': 0.4624714581496486, 'bagging_freq': 7, 'min_child_samples': 92, 'reg_alpha': 2.532169032790862, 'reg_lambda': 0.03343721523479397, 'max_depth': 17, 'min_split_gain': 0.6645943817728233, 'num_boost_round': 2255}. Best is trial 24 with value: 0.8806421596068326.
Best trial: 28. Best value: 0.880653:  58%|█████▊    | 29/50 [2:05:38<1:32:22, 263.95s/it]
[I 2025-12-29 12:39:01,697] Trial 28 finished with value: 0.8806530962987051 and parameters: {'num_leaves': 64, 'learning_rate': 0.11348889195804972, 'feature_fraction': 0.48945033823232814, 'bagging_fraction': 0.8872090749212161, 'bagging_freq': 3, 'min_child_samples': 67, 'reg_alpha': 0.0017990053685857821, 'reg_lambda': 0.14572371609604, 'max_depth': 20, 'min_split_gain': 0.8314457334022003, 'num_boost_round': 3520}. Best is trial 28 with value: 0.8806530962987051.
Best trial: 28. Best value: 0.880653:  60%|██████    | 30/50 [2:08:55<1:21:15, 243.79s/it]
[I 2025-12-29 12:42:18,441] Trial 29 finished with value: 0.8805159993852989 and parameters: {'num_leaves': 66, 'learning_rate': 0.11197757017396417, 'feature_fraction': 0.4778485030967995, 'bagging_fraction': 0.7260528626965246, 'bagging_freq': 3, 'min_child_samples': 121, 'reg_alpha': 0.0003741907699769493, 'reg_lambda': 2.52666191861327, 'max_depth': 15, 'min_split_gain': 0.7708578902014559, 'num_boost_round': 3575}. Best is trial 28 with value: 0.8806530962987051.
Best trial: 28. Best value: 0.880653:  62%|██████▏   | 31/50 [2:11:41<1:09:49, 220.48s/it]
[I 2025-12-29 12:45:04,527] Trial 30 finished with value: 0.880232938572895 and parameters: {'num_leaves': 46, 'learning_rate': 0.18964924526792346, 'feature_fraction': 0.42676655561860805, 'bagging_fraction': 0.812902681026608, 'bagging_freq': 2, 'min_child_samples': 70, 'reg_alpha': 0.0023409571156311213, 'reg_lambda': 0.27600207250470327, 'max_depth': 17, 'min_split_gain': 0.0026679612772813055, 'num_boost_round': 3647}. Best is trial 28 with value: 0.8806530962987051.
Best trial: 31. Best value: 0.880827:  64%|██████▍   | 32/50 [2:20:20<1:33:00, 310.05s/it]
[I 2025-12-29 12:53:43,585] Trial 31 finished with value: 0.8808271381161842 and parameters: {'num_leaves': 61, 'learning_rate': 0.07992723764599755, 'feature_fraction': 0.490785240652546, 'bagging_fraction': 0.9361429808092586, 'bagging_freq': 5, 'min_child_samples': 60, 'reg_alpha': 0.210483838010072, 'reg_lambda': 0.1771858833407956, 'max_depth': 20, 'min_split_gain': 0.9390933850724605, 'num_boost_round': 3401}. Best is trial 31 with value: 0.8808271381161842.
Best trial: 31. Best value: 0.880827:  66%|██████▌   | 33/50 [2:23:18<1:16:34, 270.25s/it]
[I 2025-12-29 12:56:40,972] Trial 32 finished with value: 0.8802234238316862 and parameters: {'num_leaves': 63, 'learning_rate': 0.18662252059276926, 'feature_fraction': 0.5008965792602997, 'bagging_fraction': 0.9295625398201912, 'bagging_freq': 6, 'min_child_samples': 62, 'reg_alpha': 0.2607250992156375, 'reg_lambda': 0.11572746238656921, 'max_depth': 19, 'min_split_gain': 0.8581873396810081, 'num_boost_round': 3435}. Best is trial 31 with value: 0.8808271381161842.
Best trial: 31. Best value: 0.880827:  68%|██████▊   | 34/50 [2:27:15<1:09:25, 260.32s/it]
[I 2025-12-29 13:00:38,111] Trial 33 finished with value: 0.8806061278682379 and parameters: {'num_leaves': 75, 'learning_rate': 0.1141173165787607, 'feature_fraction': 0.5524894747429165, 'bagging_fraction': 0.8696324744738877, 'bagging_freq': 5, 'min_child_samples': 99, 'reg_alpha': 0.0034416109913813307, 'reg_lambda': 1.227673686235478, 'max_depth': 20, 'min_split_gain': 0.6750300035686145, 'num_boost_round': 2899}. Best is trial 31 with value: 0.8808271381161842.
Best trial: 31. Best value: 0.880827:  70%|███████   | 35/50 [2:30:59<1:02:23, 249.56s/it]
[I 2025-12-29 13:04:22,571] Trial 34 finished with value: 0.8803244255416443 and parameters: {'num_leaves': 43, 'learning_rate': 0.0793942028610545, 'feature_fraction': 0.40114488114562863, 'bagging_fraction': 0.8710596192880172, 'bagging_freq': 3, 'min_child_samples': 76, 'reg_alpha': 0.07146965633899709, 'reg_lambda': 0.39609828234333044, 'max_depth': 18, 'min_split_gain': 0.8089569629222335, 'num_boost_round': 1253}. Best is trial 31 with value: 0.8808271381161842.
Best trial: 35. Best value: 0.8809:  72%|███████▏  | 36/50 [2:42:21<1:28:27, 379.12s/it]  
[I 2025-12-29 13:15:43,981] Trial 35 finished with value: 0.8808995828824046 and parameters: {'num_leaves': 56, 'learning_rate': 0.054435083950058305, 'feature_fraction': 0.4908067005167072, 'bagging_fraction': 0.9523096609182985, 'bagging_freq': 2, 'min_child_samples': 60, 'reg_alpha': 5.4103369772881856e-05, 'reg_lambda': 0.00018238380309166524, 'max_depth': 12, 'min_split_gain': 0.9308308448123823, 'num_boost_round': 3732}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 35. Best value: 0.8809:  74%|███████▍  | 37/50 [2:51:52<1:34:37, 436.75s/it]
[I 2025-12-29 13:25:15,214] Trial 36 finished with value: 0.880572075212246 and parameters: {'num_leaves': 58, 'learning_rate': 0.02583590162751186, 'feature_fraction': 0.4945228375856424, 'bagging_fraction': 0.9457309732993237, 'bagging_freq': 1, 'min_child_samples': 59, 'reg_alpha': 2.723438020439091e-05, 'reg_lambda': 9.170550408340204e-05, 'max_depth': 12, 'min_split_gain': 0.9165386979887824, 'num_boost_round': 3752}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 35. Best value: 0.8809:  76%|███████▌  | 38/50 [3:02:44<1:40:17, 501.49s/it]
[I 2025-12-29 13:36:07,760] Trial 37 finished with value: 0.8801375074845554 and parameters: {'num_leaves': 59, 'learning_rate': 0.014266595078681157, 'feature_fraction': 0.45582238932640595, 'bagging_fraction': 0.8969738546030452, 'bagging_freq': 2, 'min_child_samples': 108, 'reg_alpha': 0.0007842185005993474, 'reg_lambda': 1.4731700450269638e-05, 'max_depth': 13, 'min_split_gain': 0.1141624323516465, 'num_boost_round': 3475}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 35. Best value: 0.8809:  78%|███████▊  | 39/50 [3:12:29<1:36:29, 526.28s/it]
[I 2025-12-29 13:45:51,888] Trial 38 finished with value: 0.8808724255938973 and parameters: {'num_leaves': 72, 'learning_rate': 0.053382823305503396, 'feature_fraction': 0.5212563950069323, 'bagging_fraction': 0.9534004465778745, 'bagging_freq': 1, 'min_child_samples': 86, 'reg_alpha': 6.701795263007436e-05, 'reg_lambda': 0.0002761964118801679, 'max_depth': 11, 'min_split_gain': 0.2928149094443459, 'num_boost_round': 3997}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 35. Best value: 0.8809:  80%|████████  | 40/50 [3:14:14<1:06:39, 399.93s/it]
[I 2025-12-29 13:47:36,991] Trial 39 finished with value: 0.8804191017711878 and parameters: {'num_leaves': 85, 'learning_rate': 0.2324208227251834, 'feature_fraction': 0.4414709633633172, 'bagging_fraction': 0.9660017277854736, 'bagging_freq': 1, 'min_child_samples': 89, 'reg_alpha': 0.0001433882015411028, 'reg_lambda': 0.00032983465263466825, 'max_depth': 8, 'min_split_gain': 0.29336494065451774, 'num_boost_round': 3928}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 35. Best value: 0.8809:  82%|████████▏ | 41/50 [3:22:31<1:04:22, 429.13s/it]
[I 2025-12-29 13:55:54,265] Trial 40 finished with value: 0.8807463847009969 and parameters: {'num_leaves': 73, 'learning_rate': 0.061641670641417706, 'feature_fraction': 0.48922794312525936, 'bagging_fraction': 0.8459220499765501, 'bagging_freq': 2, 'min_child_samples': 37, 'reg_alpha': 1.5943636277283512e-06, 'reg_lambda': 1.7502851502562083e-07, 'max_depth': 11, 'min_split_gain': 0.7469497031967245, 'num_boost_round': 3738}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 35. Best value: 0.8809:  84%|████████▍ | 42/50 [3:30:35<59:23, 445.49s/it]  
[I 2025-12-29 14:03:57,933] Trial 41 finished with value: 0.8807924253329116 and parameters: {'num_leaves': 72, 'learning_rate': 0.053734449396892085, 'feature_fraction': 0.4860505296905124, 'bagging_fraction': 0.8921734366324267, 'bagging_freq': 2, 'min_child_samples': 54, 'reg_alpha': 4.4465637167074276e-07, 'reg_lambda': 3.219663379846475e-08, 'max_depth': 11, 'min_split_gain': 0.9521140316453215, 'num_boost_round': 3729}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 35. Best value: 0.8809:  86%|████████▌ | 43/50 [3:42:08<1:00:39, 519.99s/it]
[I 2025-12-29 14:15:31,736] Trial 42 finished with value: 0.880818932200785 and parameters: {'num_leaves': 73, 'learning_rate': 0.03928715233979115, 'feature_fraction': 0.46570630347651637, 'bagging_fraction': 0.8434051691288628, 'bagging_freq': 2, 'min_child_samples': 37, 'reg_alpha': 2.9224120090368334e-07, 'reg_lambda': 1.1274330465462936e-08, 'max_depth': 11, 'min_split_gain': 0.9403973597644752, 'num_boost_round': 3779}. Best is trial 35 with value: 0.8808995828824046.
Best trial: 43. Best value: 0.880906:  88%|████████▊ | 44/50 [3:53:20<56:32, 565.39s/it]  
[I 2025-12-29 14:26:43,056] Trial 43 finished with value: 0.8809063085197089 and parameters: {'num_leaves': 70, 'learning_rate': 0.039606279762167775, 'feature_fraction': 0.533999832662499, 'bagging_fraction': 0.960839199160537, 'bagging_freq': 2, 'min_child_samples': 58, 'reg_alpha': 3.2503292052709743e-07, 'reg_lambda': 1.1006812990479833e-08, 'max_depth': 9, 'min_split_gain': 0.9463807224250697, 'num_boost_round': 3769}. Best is trial 43 with value: 0.8809063085197089.
Best trial: 44. Best value: 0.880994:  90%|█████████ | 45/50 [4:03:29<48:12, 578.60s/it]
[I 2025-12-29 14:36:52,474] Trial 44 finished with value: 0.8809935551799978 and parameters: {'num_leaves': 81, 'learning_rate': 0.037634969446578445, 'feature_fraction': 0.5379351549326781, 'bagging_fraction': 0.96149910340915, 'bagging_freq': 1, 'min_child_samples': 40, 'reg_alpha': 1.7501119220189e-07, 'reg_lambda': 4.1337549895704806e-07, 'max_depth': 8, 'min_split_gain': 0.9365273083034507, 'num_boost_round': 3901}. Best is trial 44 with value: 0.8809935551799978.
Best trial: 44. Best value: 0.880994:  92%|█████████▏| 46/50 [4:13:44<39:18, 589.59s/it]
[I 2025-12-29 14:47:07,718] Trial 45 finished with value: 0.880863090816566 and parameters: {'num_leaves': 81, 'learning_rate': 0.0261339733757742, 'feature_fraction': 0.5343802193592775, 'bagging_fraction': 0.9639016089694906, 'bagging_freq': 1, 'min_child_samples': 68, 'reg_alpha': 7.106211224645722e-08, 'reg_lambda': 8.64736474372071e-07, 'max_depth': 8, 'min_split_gain': 0.19434294379882422, 'num_boost_round': 3908}. Best is trial 44 with value: 0.8809935551799978.
Best trial: 44. Best value: 0.880994:  94%|█████████▍| 47/50 [4:24:07<29:58, 599.55s/it]
[I 2025-12-29 14:57:30,506] Trial 46 finished with value: 0.8808616602212342 and parameters: {'num_leaves': 83, 'learning_rate': 0.026424770421984663, 'feature_fraction': 0.6223349748005929, 'bagging_fraction': 0.9664182917445291, 'bagging_freq': 1, 'min_child_samples': 70, 'reg_alpha': 3.9463082459179665e-08, 'reg_lambda': 5.46881315316439e-07, 'max_depth': 8, 'min_split_gain': 0.23640275140987585, 'num_boost_round': 4000}. Best is trial 44 with value: 0.8809935551799978.
Best trial: 44. Best value: 0.880994:  96%|█████████▌| 48/50 [4:34:14<20:03, 601.81s/it]
[I 2025-12-29 15:07:37,600] Trial 47 finished with value: 0.8806254802886171 and parameters: {'num_leaves': 80, 'learning_rate': 0.018736678863240337, 'feature_fraction': 0.5371641462388899, 'bagging_fraction': 0.9732242245872018, 'bagging_freq': 1, 'min_child_samples': 85, 'reg_alpha': 1.478316833055122e-07, 'reg_lambda': 4.0762492824734955e-06, 'max_depth': 9, 'min_split_gain': 0.18004713079372064, 'num_boost_round': 3819}. Best is trial 44 with value: 0.8809935551799978.
Best trial: 48. Best value: 0.881015:  98%|█████████▊| 49/50 [4:44:08<09:59, 599.34s/it]
[I 2025-12-29 15:17:31,161] Trial 48 finished with value: 0.881014761917504 and parameters: {'num_leaves': 93, 'learning_rate': 0.040068149332407606, 'feature_fraction': 0.5770105839189245, 'bagging_fraction': 0.9061258968947112, 'bagging_freq': 1, 'min_child_samples': 97, 'reg_alpha': 1.2427700560713473e-08, 'reg_lambda': 1.1808151399114223e-07, 'max_depth': 7, 'min_split_gain': 0.3532569422198928, 'num_boost_round': 3636}. Best is trial 48 with value: 0.881014761917504.
Best trial: 48. Best value: 0.881015: 100%|██████████| 50/50 [4:52:36<00:00, 351.14s/it]
[I 2025-12-29 15:25:59,628] Trial 49 finished with value: 0.8809017045538727 and parameters: {'num_leaves': 91, 'learning_rate': 0.03893063501646156, 'feature_fraction': 0.570176936042701, 'bagging_fraction': 0.9988774832158618, 'bagging_freq': 1, 'min_child_samples': 119, 'reg_alpha': 1.1625811616269783e-08, 'reg_lambda': 9.510742255175905e-08, 'max_depth': 7, 'min_split_gain': 0.32324344448046527, 'num_boost_round': 3280}. Best is trial 48 with value: 0.881014761917504.
результат оптимизации:
Лучший ROC-AUC: 0.881015
Лучшие параметры:
  num_leaves: 93
  learning_rate: 0.040068149332407606
  feature_fraction: 0.5770105839189245
  bagging_fraction: 0.9061258968947112
  bagging_freq: 1
  min_child_samples: 97
  reg_alpha: 1.2427700560713473e-08
  reg_lambda: 1.1808151399114223e-07
  max_depth: 7
  min_split_gain: 0.3532569422198928
  num_boost_round: 3636
============================================================


Обучение финальной модели на полном очищенном датасете
Training until validation scores don't improve for 150 rounds
[200]	valid_0's auc: 0.874368
[400]	valid_0's auc: 0.877677
[600]	valid_0's auc: 0.87895
[800]	valid_0's auc: 0.87953
[1000]	valid_0's auc: 0.879855
[1200]	valid_0's auc: 0.880109
[1400]	valid_0's auc: 0.880293
[1600]	valid_0's auc: 0.880423
[1800]	valid_0's auc: 0.880519
[2000]	valid_0's auc: 0.880589
[2200]	valid_0's auc: 0.880659
[2400]	valid_0's auc: 0.880714
[2600]	valid_0's auc: 0.880755
[2800]	valid_0's auc: 0.880786
[3000]	valid_0's auc: 0.88081
[3200]	valid_0's auc: 0.880833
[3400]	valid_0's auc: 0.880851
[3600]	valid_0's auc: 0.880868
Did not meet early stopping. Best iteration is:
[3636]	valid_0's auc: 0.88087
ROC-AUC на валидации: 0.880870
Модель сохранена: src\models\model_improved_optuna.pkl

Результат с оптимизацией по num_boost_round (количество деревьев): `0.88087`, получился чуть лучше чем просто 1000 для оптимизации и 3000 для финальной модели: `0.88055`

### 6.4. Простой CatBoost подход

Пробуем CatBoost с минимальной предобработкой, потому что в задачах с категориальными признаками он часто сразу дает сильный результат. Такой шаг нужен, чтобы проверить насколько CatBoost выигрывает сам по себе, без сложных пайплайнов.

#### Вспомогательная функция для CatBoost


```python
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

def simple_preprocess_catboost(X_train, X_val):
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()
    
    if 'Annual_Premium' in X_train_proc.columns:
        X_train_proc['Annual_Premium'] = X_train_proc['Annual_Premium'].fillna(0).astype('int32')
        X_val_proc['Annual_Premium'] = X_val_proc['Annual_Premium'].fillna(0).astype('int32')
    
    num_cols = [col for col in X_train_proc.columns if X_train_proc[col].dtype in ['int64', 'float64']]
    num_cols = [col for col in num_cols if col != 'Annual_Premium']
    
    for col in num_cols:
        median_val = X_train_proc[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_train_proc[col] = X_train_proc[col].fillna(median_val).astype('int16')
        X_val_proc[col] = X_val_proc[col].fillna(median_val).astype('int16')
    
    return X_train_proc, X_val_proc
```


```python
print("Обучение простого CatBoost")

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

X_train_proc_simple, X_val_proc_simple = simple_preprocess_catboost(X_train, X_val)
cat_features = X_train_proc_simple.columns.values

sample_size = 1000000
if len(X_train_proc_simple) > sample_size:
    sample_idx = np.random.choice(len(X_train_proc_simple), sample_size, replace=False)
    X_train_sample = X_train_proc_simple.iloc[sample_idx]
    y_train_sample = y_train.iloc[sample_idx]
    print(f"Используется выборка из {sample_size:,} строк")
else:
    X_train_sample = X_train_proc_simple
    y_train_sample = y_train

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.03,
    'iterations': 3000,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'bagging_temperature': 1.0,
    'border_count': 254,
    'colsample_bylevel': 1.0,
    'min_data_in_leaf': 1,
    'random_seed': 42,
    'task_type': 'GPU',
    'gpu_ram_part': 0.9,
    'verbose': False
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
models_simple = []
oof_predictions = np.zeros(len(X_train_sample))

print(f"Обучение {n_folds} фолдов")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_sample, y_train_sample)):
    print(f"  Fold {fold+1}/{n_folds}")
    X_tr, X_te = X_train_sample.iloc[train_idx], X_train_sample.iloc[val_idx]
    y_tr, y_te = y_train_sample.iloc[train_idx], y_train_sample.iloc[val_idx]
    
    X_tr_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    X_te_pool = Pool(X_te, y_te, cat_features=cat_features)
    
    model = CatBoostClassifier(**cat_params)
    model.fit(X_tr_pool, eval_set=X_te_pool, verbose=1000, early_stopping_rounds=200)
    
    oof_predictions[val_idx] = model.predict_proba(X_te_pool)[:, 1]
    models_simple.append(model)

cv_score_simple = roc_auc_score(y_train_sample, oof_predictions)
print(f"CV ROC-AUC: {cv_score_simple:.6f}")

X_val_pool = Pool(X_val_proc_simple, cat_features=cat_features)
val_predictions_simple = np.mean([m.predict_proba(X_val_pool)[:, 1] for m in models_simple], axis=0)
score_simple = roc_auc_score(y_val, val_predictions_simple)
print(f"ROC-AUC на валидации: {score_simple:.6f}")

joblib.dump({'models': models_simple, 'cat_features': cat_features}, MODELS_DIR / 'catboost_simple_final.pkl')
print(f"Модель сохранена: {MODELS_DIR / 'catboost_simple_final.pkl'}")
```

    Обучение простого CatBoost
    Используется выборка из 1,000,000 строк
    Обучение 5 фолдов
      Fold 1/5
    

    Default metric period is 5 because AUC is/are not implemented for GPU
    

    0:	test: 0.8616198	best: 0.8616198 (0)	total: 54.1ms	remaining: 2m 42s
    1000:	test: 0.8860235	best: 0.8860244 (998)	total: 25.3s	remaining: 50.6s
    2000:	test: 0.8865267	best: 0.8865294 (1989)	total: 48s	remaining: 24s
    2999:	test: 0.8866575	best: 0.8866630 (2977)	total: 1m 10s	remaining: 0us
    bestTest = 0.8866629601
    bestIteration = 2977
    Shrink model to first 2978 iterations.
      Fold 2/5
    

    Default metric period is 5 because AUC is/are not implemented for GPU
    

    0:	test: 0.8614763	best: 0.8614763 (0)	total: 34.8ms	remaining: 1m 44s
    1000:	test: 0.8864266	best: 0.8864266 (1000)	total: 23.9s	remaining: 47.8s
    2000:	test: 0.8868688	best: 0.8868691 (1993)	total: 47.1s	remaining: 23.5s
    2999:	test: 0.8870326	best: 0.8870361 (2963)	total: 1m 10s	remaining: 0us
    bestTest = 0.8870360851
    bestIteration = 2963
    Shrink model to first 2964 iterations.
      Fold 3/5
    

    Default metric period is 5 because AUC is/are not implemented for GPU
    

    0:	test: 0.8607248	best: 0.8607248 (0)	total: 26.8ms	remaining: 1m 20s
    1000:	test: 0.8865068	best: 0.8865068 (1000)	total: 23.7s	remaining: 47.3s
    2000:	test: 0.8869871	best: 0.8869874 (1999)	total: 47s	remaining: 23.5s
    2999:	test: 0.8871457	best: 0.8871462 (2997)	total: 1m 9s	remaining: 0us
    bestTest = 0.8871462345
    bestIteration = 2997
    Shrink model to first 2998 iterations.
      Fold 4/5
    

    Default metric period is 5 because AUC is/are not implemented for GPU
    

    0:	test: 0.8609999	best: 0.8609999 (0)	total: 37.5ms	remaining: 1m 52s
    1000:	test: 0.8863213	best: 0.8863213 (1000)	total: 24.2s	remaining: 48.3s
    2000:	test: 0.8869185	best: 0.8869185 (2000)	total: 47.7s	remaining: 23.8s
    2999:	test: 0.8871346	best: 0.8871393 (2981)	total: 1m 10s	remaining: 0us
    bestTest = 0.8871392608
    bestIteration = 2981
    Shrink model to first 2982 iterations.
      Fold 5/5
    

    Default metric period is 5 because AUC is/are not implemented for GPU
    

    0:	test: 0.8619809	best: 0.8619809 (0)	total: 37.2ms	remaining: 1m 51s
    1000:	test: 0.8859610	best: 0.8859610 (1000)	total: 23.3s	remaining: 46.5s
    2000:	test: 0.8863873	best: 0.8863873 (2000)	total: 46.2s	remaining: 23s
    2999:	test: 0.8865442	best: 0.8865442 (2999)	total: 1m 8s	remaining: 0us
    bestTest = 0.8865442276
    bestIteration = 2999
    CV ROC-AUC: 0.886898
    ROC-AUC на валидации: 0.887494
    Модель сохранена: src\models\catboost_simple_final.pkl
    

Принципиально более интересный результат `0.88749` у cb благодаря хорошей работе с категориальными признаками, намного превосходит lgbm `0.88055` и `0.87945` у LightAutoML, будем продолжать работу над катбустом. При этом немного быстрее по времени чем другие методы

### 6.5. Продвинутый CatBoost с feature engineering

На этом шаге специально усложняем подготовку данных и добавляем новые признаки, чтобы проверить гипотезу из EDA, что дополнительные взаимодействия могут поднять качество. Чистим дубликаты и часть выбросов, добавляем несколько новых признаков и снова обучаем CatBoost.


```python
model_path = MODELS_DIR / 'catboost_advanced_final.pkl'

def remove_duplicates_advanced(X, y=None):
    initial_size = len(X)
    if y is not None:
        X_with_target = X.copy()
        X_with_target['_target'] = y.values
        X_with_target = X_with_target.drop_duplicates()
        y_cleaned = pd.Series(X_with_target['_target'].values, index=X_with_target.index)
        X_cleaned = X_with_target.drop('_target', axis=1)
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned, y_cleaned
    else:
        X_cleaned = X.drop_duplicates()
        duplicates_removed = initial_size - len(X_cleaned)
        if duplicates_removed > 0:
            print(f"Удалено дубликатов: {duplicates_removed:,} ({duplicates_removed/initial_size*100:.2f}%)")
        return X_cleaned

def detect_and_remove_anomalies_advanced(X, y=None, z_threshold=3.5, iqr_factor=2.0):
    initial_size = len(X)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_indices = set()
    
    for col in numeric_cols:
        col_data = X[col].dropna()
        if len(col_data) == 0:
            continue
        z_scores = np.abs(stats.zscore(col_data))
        z_outliers = col_data.index[z_scores > z_threshold].tolist()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            iqr_outliers = col_data.index[(col_data < lower_bound) | (col_data > upper_bound)].tolist()
        else:
            iqr_outliers = []
        outliers = set(z_outliers) | set(iqr_outliers)
        anomaly_indices.update(outliers)
    
    if len(anomaly_indices) > 0:
        anomaly_pct = len(anomaly_indices) / initial_size * 100
        if anomaly_pct < 8.0:
            X_cleaned = X.drop(index=anomaly_indices)
            if y is not None:
                y_cleaned = y.drop(index=anomaly_indices)
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned, y_cleaned
            else:
                print(f"Удалено аномалий: {len(anomaly_indices):,} ({anomaly_pct:.2f}%)")
                return X_cleaned
    
    if y is not None:
        return X, y
    else:
        return X

def create_features_advanced(df, is_train=False):
    df = df.copy()
    if 'Annual_Premium' in df.columns:
        df['Annual_Premium_log'] = np.log1p(df['Annual_Premium'])
        df['Annual_Premium_sqrt'] = np.sqrt(df['Annual_Premium'])
    if 'Age' in df.columns:
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['<25', '25-35', '35-45', '45-55', '55+'])
        df['Age_group'] = df['Age_group'].astype(str)
        df['Age_squared'] = df['Age'] ** 2
        df['Age_cubed'] = df['Age'] ** 3
    if 'Vintage' in df.columns:
        df['Vintage_log'] = np.log1p(df['Vintage'])
    if 'Age' in df.columns and 'Annual_Premium' in df.columns:
        df['Age_Premium_interaction'] = df['Age'] * df['Annual_Premium'] / 1000
        df['Age_Premium_ratio'] = df['Annual_Premium'] / (df['Age'] + 1)
    if 'Vehicle_Age' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Vehicle_combination'] = df['Vehicle_Age'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Previously_Insured' in df.columns and 'Vehicle_Damage' in df.columns:
        df['Insured_Damage'] = df['Previously_Insured'].astype(str) + '_' + df['Vehicle_Damage'].astype(str)
    if 'Region_Code' in df.columns and 'Policy_Sales_Channel' in df.columns:
        df['Region_Channel'] = df['Region_Code'].astype(str) + '_' + df['Policy_Sales_Channel'].astype(str)
    return df

def preprocess_data_advanced(X_train, X_val):
    X_train_proc = create_features_advanced(X_train, is_train=True)
    X_val_proc = create_features_advanced(X_val, is_train=False)
    
    if 'Annual_Premium' in X_train_proc.columns:
        X_train_proc['Annual_Premium'] = X_train_proc['Annual_Premium'].fillna(0).astype('int32')
        X_val_proc['Annual_Premium'] = X_val_proc['Annual_Premium'].fillna(0).astype('int32')
    
    num_cols = [col for col in X_train_proc.columns if X_train_proc[col].dtype in ['int64', 'float64']]
    num_cols = [col for col in num_cols if col != 'Annual_Premium']
    
    for col in num_cols:
        median_val = X_train_proc[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_train_proc[col] = X_train_proc[col].fillna(median_val).astype('int16')
        X_val_proc[col] = X_val_proc[col].fillna(median_val).astype('int16')
    
    cat_features = [col for col in X_train_proc.columns if X_train_proc[col].dtype == 'object']
    
    return X_train_proc, X_val_proc, cat_features

print("Обучение продвинутого CatBoost")

X_train_clean, y_train_clean = remove_duplicates_advanced(X_train.copy(), y_train.copy())
X_train_clean, y_train_clean = detect_and_remove_anomalies_advanced(X_train_clean, y_train_clean)
X_train_proc_adv, X_val_proc_adv, cat_features_adv = preprocess_data_advanced(X_train_clean, X_val.copy())

sample_size = 1000000
if len(X_train_proc_adv) > sample_size:
    sample_idx = np.random.choice(len(X_train_proc_adv), sample_size, replace=False)
    X_train_sample = X_train_proc_adv.iloc[sample_idx]
    y_train_sample = y_train_clean.iloc[sample_idx]
    print(f"Используется выборка из {sample_size:,} строк")
else:
    X_train_sample = X_train_proc_adv
    y_train_sample = y_train_clean

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.03,
    'iterations': 3000,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'bagging_temperature': 1.0,
    'border_count': 254,
    'colsample_bylevel': 1.0,
    'min_data_in_leaf': 1,
    'random_seed': 42,
    'verbose': False
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
models_advanced = []
oof_predictions = np.zeros(len(X_train_sample))

print(f"Обучение {n_folds} фолдов")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_sample, y_train_sample)):
    print(f"  Fold {fold+1}/{n_folds}")
    X_tr, X_te = X_train_sample.iloc[train_idx], X_train_sample.iloc[val_idx]
    y_tr, y_te = y_train_sample.iloc[train_idx], y_train_sample.iloc[val_idx]
    
    X_tr_pool = Pool(X_tr, y_tr, cat_features=cat_features_adv)
    X_te_pool = Pool(X_te, y_te, cat_features=cat_features_adv)
    
    model = CatBoostClassifier(**cat_params)
    model.fit(X_tr_pool, eval_set=X_te_pool, verbose=1000, early_stopping_rounds=200)
    
    oof_predictions[val_idx] = model.predict_proba(X_te_pool)[:, 1]
    models_advanced.append(model)

cv_score_advanced = roc_auc_score(y_train_sample, oof_predictions)
print(f"CV ROC-AUC: {cv_score_advanced:.6f}")

X_val_pool_adv = Pool(X_val_proc_adv, cat_features=cat_features_adv)
val_predictions_advanced = np.mean([m.predict_proba(X_val_pool_adv)[:, 1] for m in models_advanced], axis=0)
score_advanced = roc_auc_score(y_val, val_predictions_advanced)
print(f"ROC-AUC на валидации: {score_advanced:.6f}")

joblib.dump({'models': models_advanced, 'cat_features': cat_features_adv}, MODELS_DIR / 'catboost_advanced_final.pkl')
print(f"Модель сохранена: {MODELS_DIR / 'catboost_advanced_final.pkl'}")
```

    Обучение продвинутого CatBoost
    Удалено аномалий: 95,939 (1.04%)
    Используется выборка из 800,000 строк
    Обучение 5 фолдов
      Fold 1/5
    0:	test: 0.8279962	best: 0.8279962 (0)	total: 218ms	remaining: 10m 53s
    1000:	test: 0.8720287	best: 0.8720287 (1000)	total: 1m 49s	remaining: 3m 38s
    2000:	test: 0.8741949	best: 0.8741949 (2000)	total: 3m 45s	remaining: 1m 52s
    2999:	test: 0.8749932	best: 0.8749932 (2999)	total: 5m 43s	remaining: 0us
    
    bestTest = 0.8749931853
    bestIteration = 2999
    
      Fold 2/5
    0:	test: 0.8430991	best: 0.8430991 (0)	total: 127ms	remaining: 6m 22s
    1000:	test: 0.8766117	best: 0.8766123 (996)	total: 1m 51s	remaining: 3m 43s
    2000:	test: 0.8785238	best: 0.8785239 (1998)	total: 3m 48s	remaining: 1m 54s
    2999:	test: 0.8792830	best: 0.8792830 (2999)	total: 5m 46s	remaining: 0us
    
    bestTest = 0.8792830024
    bestIteration = 2999
    
      Fold 3/5
    0:	test: 0.8419311	best: 0.8419311 (0)	total: 135ms	remaining: 6m 45s
    1000:	test: 0.8753691	best: 0.8753693 (999)	total: 1m 52s	remaining: 3m 45s
    2000:	test: 0.8776789	best: 0.8776789 (2000)	total: 3m 50s	remaining: 1m 54s
    2999:	test: 0.8785685	best: 0.8785685 (2999)	total: 5m 47s	remaining: 0us
    
    bestTest = 0.8785685141
    bestIteration = 2999
    
      Fold 4/5
    0:	test: 0.8389050	best: 0.8389050 (0)	total: 132ms	remaining: 6m 34s
    1000:	test: 0.8736826	best: 0.8736826 (1000)	total: 1m 48s	remaining: 3m 37s
    2000:	test: 0.8760144	best: 0.8760144 (2000)	total: 3m 41s	remaining: 1m 50s
    2999:	test: 0.8768385	best: 0.8768385 (2999)	total: 5m 34s	remaining: 0us
    
    bestTest = 0.8768384712
    bestIteration = 2999
    
      Fold 5/5
    0:	test: 0.8405894	best: 0.8405894 (0)	total: 142ms	remaining: 7m 6s
    1000:	test: 0.8726445	best: 0.8726445 (1000)	total: 1m 48s	remaining: 3m 37s
    2000:	test: 0.8749744	best: 0.8749744 (2000)	total: 3m 42s	remaining: 1m 51s
    2999:	test: 0.8758036	best: 0.8758054 (2991)	total: 5m 41s	remaining: 0us
    
    bestTest = 0.8758053862
    bestIteration = 2991
    
    Shrink model to first 2992 iterations.
    CV ROC-AUC: 0.877092
    ROC-AUC на валидации: 0.876462
    Модель сохранена: src\catboost_advanced_final.pkl
    

 Из результата `0.87646` против `0.88749` видно что результаты ухудшились, катбусту достаточно имеющихся признаков и простой предобработки, поэтому дальше работаем над простым вариантом

### 6.6. CatBoost с оптимизацией гиперпараметров через Optuna

Здесь оставляем минимальную предобработку как в простом CatBoost, но подбираем параметры автоматически через Optuna. Оптимизацию делаем на меньшей выборке, чтобы это занимало разумное время, а потом используем найденные настройки для финального обучения. Идея простая, если качество еще можно улучшить, то чаще всего это делается настройкой модели, а не добавлением новых фич.


```python
sample_size = 6000000
used_ram_limit = '24GB'

import optuna
from optuna.samplers import TPESampler
import json
import numpy as np
from pathlib import Path
import joblib
import gc

print("Обучение CatBoost с Optuna")

X_train_proc_optuna, X_val_proc_optuna = simple_preprocess_catboost(X_train, X_val)
cat_features_optuna = X_train_proc_optuna.columns.values

params_file = PARAMS_DIR / 'catboost_optuna_params.json'
best_params = None
best_auc_optuna = None

if params_file.exists():
    print(f"Загрузка сохраненных параметров из {params_file}")
    with open(params_file, 'r') as f:
        saved_data = json.load(f)
        best_params = saved_data.get('best_params')
        best_auc_optuna = saved_data.get('best_auc')
        print(f"Загружен AUC: {best_auc_optuna:.6f}")
else:
    print("Сохраненных параметров не найдено, запускаем оптимизацию")
    
    optuna_sample_size = min(500000, len(X_train_proc_optuna))
    optuna_idx = np.random.choice(len(X_train_proc_optuna), optuna_sample_size, replace=False)
    X_optuna = X_train_proc_optuna.iloc[optuna_idx]
    y_optuna = y_train.iloc[optuna_idx]
    print(f"Для оптимизации используется выборка {optuna_sample_size:,} строк")

    def objective(trial):
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
        
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'iterations': trial.suggest_int('iterations', 2000, 5000),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bootstrap_type': bootstrap_type,
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
            'task_type': 'GPU',
            'random_seed': 42,
            'verbose': False
        }
        
        params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.6, 1.0)
        
        if bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        else:
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_optuna, y_optuna):
            X_tr, X_v = X_optuna.iloc[train_idx], X_optuna.iloc[val_idx]
            y_tr, y_v = y_optuna.iloc[train_idx], y_optuna.iloc[val_idx]
            
            X_tr_pool = Pool(X_tr, y_tr, cat_features=cat_features_optuna)
            X_v_pool = Pool(X_v, y_v, cat_features=cat_features_optuna)
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr_pool, eval_set=X_v_pool, verbose=False, early_stopping_rounds=200)
            
            y_pred = model.predict_proba(X_v_pool)[:, 1]
            scores.append(roc_auc_score(y_v, y_pred))
        
        return np.mean(scores)

    print("Оптимизация гиперпараметров")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_auc_optuna = study.best_value
    print(f"Лучший AUC: {best_auc_optuna:.6f}")
    print(f"Лучшие параметры: {best_params}")
    
    params_data = {
        'best_params': best_params,
        'best_auc': best_auc_optuna
    }
    with open(params_file, 'w') as f:
        json.dump(params_data, f, indent=2)
    print(f"Параметры сохранены")

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': False,
    'task_type': 'GPU',
    'gpu_ram_part': 0.95,
    'used_ram_limit': used_ram_limit,
    **best_params
}

folds_dir = MODELS_DIR / 'catboost_folds'
folds_dir.mkdir(parents=True, exist_ok=True)

sample_idx_path = folds_dir / 'sample_idx.npy'

if sample_idx_path.exists():
    print(f"Загрузка индексов выборки из {sample_idx_path}")
    sample_idx = np.load(sample_idx_path)
    X_train_sample = X_train_proc_optuna.iloc[sample_idx]
    y_train_sample = y_train.iloc[sample_idx]
    print(f"Используется сохраненная выборка из {len(X_train_sample):,} строк")
else:
    if len(X_train_proc_optuna) > sample_size:
        sample_idx = np.random.choice(len(X_train_proc_optuna), sample_size, replace=False)
        X_train_sample = X_train_proc_optuna.iloc[sample_idx]
        y_train_sample = y_train.iloc[sample_idx]
        np.save(sample_idx_path, sample_idx)
        print(f"Используется новая выборка из {sample_size:,} строк")
    else:
        X_train_sample = X_train_proc_optuna
        y_train_sample = y_train
        sample_idx = np.arange(len(X_train_sample))
        np.save(sample_idx_path, sample_idx)
        print(f"Используется полная выборка из {len(X_train_sample):,} строк")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X_train_sample))

trained_folds_path = folds_dir / 'trained_folds.npy'
oof_path = folds_dir / 'oof_predictions.npy'

trained_folds = []
if trained_folds_path.exists():
    trained_folds = np.load(trained_folds_path).tolist()
    print(f"Найдено {len(trained_folds)} уже обученных фолдов: {trained_folds}")
    if oof_path.exists():
        oof_predictions = np.load(oof_path)

print(f"Обучение {n_folds} фолдов")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_sample, y_train_sample)):
    if fold in trained_folds:
        print(f"Fold {fold+1}/{n_folds} пропущен (уже обучен)")
        continue

    print(f"Fold {fold+1}/{n_folds}")
    X_tr, X_te = X_train_sample.iloc[train_idx], X_train_sample.iloc[val_idx]
    y_tr, y_te = y_train_sample.iloc[train_idx], y_train_sample.iloc[val_idx]
    
    X_tr_pool = Pool(X_tr, y_tr, cat_features=cat_features_optuna)
    X_te_pool = Pool(X_te, y_te, cat_features=cat_features_optuna)
    
    model = CatBoostClassifier(**cat_params)
    model.fit(X_tr_pool, eval_set=X_te_pool, verbose=1000, early_stopping_rounds=200)
    
    model_path = folds_dir / f'catboost_fold_{fold}.cbm'
    print(f"Сохранение модели")
    model.save_model(str(model_path))
    print(f"Модель сохранена в {model_path}")
    
    print(f"Получение предсказаний")
    oof_predictions[val_idx] = model.predict_proba(X_te_pool)[:, 1]
    
    del model
    del X_tr_pool, X_te_pool
    del X_tr, X_te, y_tr, y_te
    gc.collect()
    
    trained_folds.append(fold)
    np.save(oof_path, oof_predictions)
    np.save(trained_folds_path, np.array(trained_folds, dtype=int))
    print(f"  Fold {fold+1}/{n_folds} завершен")

cv_score_optuna = roc_auc_score(y_train_sample, oof_predictions)
print(f"CV ROC-AUC: {cv_score_optuna:.6f}")

print("Загрузка моделей для предсказаний на валидации")
models_optuna = []
for fold in trained_folds:
    model_path_cbm = folds_dir / f'catboost_fold_{fold}.cbm'
    if model_path_cbm.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path_cbm))
        models_optuna.append(model)

X_val_pool_optuna = Pool(X_val_proc_optuna, cat_features=cat_features_optuna)
val_predictions_optuna = np.mean([m.predict_proba(X_val_pool_optuna)[:, 1] for m in models_optuna], axis=0)
score_optuna = roc_auc_score(y_val, val_predictions_optuna)
print(f"ROC-AUC на валидации: {score_optuna:.6f}")

del models_optuna, X_val_pool_optuna, val_predictions_optuna
gc.collect()

joblib.dump(
    {
        'model_paths': [str(folds_dir / f'catboost_fold_{i}.cbm') for i in trained_folds],
        'cat_features': cat_features_optuna,
        'best_params': best_params
    },
    MODELS_DIR / 'catboost_optuna_models.pkl'
)
with open(PARAMS_DIR / 'catboost_optuna_params.json', 'w') as f:
    json.dump({'best_params': best_params, 'best_auc': best_auc_optuna, 'cv_auc': cv_score_optuna}, f, indent=2)
print(f"Параметры и метаданные сохранены")
```

    Обучение CatBoost с Optuna
    Загрузка сохраненных параметров из src\params\catboost_optuna_params.json
    Загружен AUC: 0.882380
    Загрузка индексов выборки из src\models\catboost_folds\sample_idx.npy
    Используется сохраненная выборка из 6,000,000 строк
    Найдено 4 уже обученных фолдов: [0, 1, 2, 3]
    Обучение 5 фолдов
    Fold 1/5 пропущен (уже обучен)
    Fold 2/5 пропущен (уже обучен)
    Fold 3/5 пропущен (уже обучен)
    Fold 4/5 пропущен (уже обучен)
    Fold 5/5
    

    Default metric period is 5 because AUC is/are not implemented for GPU
    

    0:	test: 0.8643126	best: 0.8643126 (0)	total: 427ms	remaining: 17m 32s
    1000:	test: 0.8924683	best: 0.8924683 (1000)	total: 3m 18s	remaining: 4m 50s
    2000:	test: 0.8929046	best: 0.8929046 (1998)	total: 5m 49s	remaining: 1m 21s
    2465:	test: 0.8929967	best: 0.8929967 (2465)	total: 6m 58s	remaining: 0us
    bestTest = 0.8929966688
    bestIteration = 2465
    Сохранение модели
    Модель сохранена в src\models\catboost_folds\catboost_fold_4.cbm
    Получение предсказаний
      Fold 5/5 завершен
    CV ROC-AUC: 0.893087
    Загрузка моделей для предсказаний на валидации
    ROC-AUC на валидации: 0.893609
    Параметры и метаданные сохранены
    

### 6.7. Генерация submission для лучшей модели

Делаем предсказания для test тем же способом, что и при валидации, и сохраняем файл для Kaggle.


```python
print("Загрузка модели для submission")

folds_dir = MODELS_DIR / 'catboost_folds'
model_path = MODELS_DIR / 'catboost_optuna_models.pkl'

if model_path.exists():
    try:
        print(f"Загрузка модели из {model_path}")
        saved_data = joblib.load(model_path)
        if 'models' in saved_data:
            models_optuna = saved_data['models']
            cat_features_optuna = saved_data['cat_features']
            print(f"Загружено {len(models_optuna)} фолдов из .pkl")
        elif 'model_paths' in saved_data:
            models_optuna = []
            for model_path_str in saved_data['model_paths']:
                model = CatBoostClassifier()
                model.load_model(model_path_str)
                models_optuna.append(model)
            cat_features_optuna = saved_data['cat_features']
            print(f"Загружено {len(models_optuna)} фолдов из .cbm файлов")
        else:
            raise ValueError("Неизвестный формат сохраненных данных")
    except Exception as e:
        print(f"Ошибка загрузки из .pkl: {e}, загружаем из .cbm файлов")
        trained_folds_path = folds_dir / 'trained_folds.npy'
        if trained_folds_path.exists():
            trained_folds = np.load(trained_folds_path).tolist()
            models_optuna = []
            for fold in trained_folds:
                model_path_cbm = folds_dir / f'catboost_fold_{fold}.cbm'
                if model_path_cbm.exists():
                    model = CatBoostClassifier()
                    model.load_model(str(model_path_cbm))
                    models_optuna.append(model)
            cat_features_optuna = None
            print(f"Загружено {len(models_optuna)} фолдов из .cbm файлов")
        else:
            raise FileNotFoundError(f"Не найдены обученные фолды в {folds_dir}")
else:
    print(f"Файл {model_path} не найден, загружаем из .cbm файлов")
    trained_folds_path = folds_dir / 'trained_folds.npy'
    if trained_folds_path.exists():
        trained_folds = np.load(trained_folds_path).tolist()
        models_optuna = []
        for fold in trained_folds:
            model_path_cbm = folds_dir / f'catboost_fold_{fold}.cbm'
            if model_path_cbm.exists():
                model = CatBoostClassifier()
                model.load_model(str(model_path_cbm))
                models_optuna.append(model)
        print(f"Загружено {len(models_optuna)} фолдов из .cbm файлов")
        cat_features_optuna = None
    else:
        raise FileNotFoundError(f"Не найдены обученные фолды в {folds_dir}")

if len(models_optuna) == 0:
    raise FileNotFoundError("Не удалось загрузить модели")

print("Генерация submission")

test_data_full = pd.read_csv(DATA_DIR / 'test.csv')
print(f"Размер тестовой выборки: {len(test_data_full):,}")

X_test = test_data_full.drop(['id'], axis=1, errors='ignore')

train_sample_for_preprocess = pd.read_csv(DATA_DIR / 'train.csv').head(10000)
X_train_sample = train_sample_for_preprocess.drop(['Response', 'id'], axis=1, errors='ignore')

X_test_proc, _ = simple_preprocess_catboost(X_test, X_train_sample)

if cat_features_optuna is None:
    cat_features_optuna = X_test_proc.columns.values

X_test_pool = Pool(X_test_proc, cat_features=cat_features_optuna)

print("Предсказания на test данных")

batch_size = 100000
test_pred_proba = np.zeros(len(X_test_proc))

for i in range(0, len(X_test_proc), batch_size):
    end_idx = min(i + batch_size, len(X_test_proc))
    batch = X_test_proc.iloc[i:end_idx]
    batch_pool = Pool(batch, cat_features=cat_features_optuna)
    
    batch_preds = np.mean([m.predict_proba(batch_pool)[:, 1] for m in models_optuna], axis=0)
    test_pred_proba[i:end_idx] = batch_preds
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"  Обработано {end_idx:,} / {len(X_test_proc):,} строк")

submission_custom = pd.DataFrame({
    'id': test_data_full['id'].values,
    'Response': test_pred_proba
})

submission_file_notebook = SUBMIT_DIR / 'custom_catboost_optuna_submission.csv'
submission_custom.to_csv(submission_file_notebook, index=False)

print(f"Submission сохранен: {submission_file_notebook}")
print(f"Размер: {submission_custom.shape}")
print(f"Диапазон предсказаний: [{test_pred_proba.min():.6f}, {test_pred_proba.max():.6f}]")
print(f"Среднее предсказание: {test_pred_proba.mean():.6f}")

```

    Загрузка модели для submission
    Загрузка модели из src\models\catboost_optuna_models.pkl
    Загружено 5 фолдов из .cbm файлов
    Генерация submission
    Размер тестовой выборки: 7,669,866
    Предсказания на test данных
      Обработано 1,000,000 / 7,669,866 строк
      Обработано 2,000,000 / 7,669,866 строк
      Обработано 3,000,000 / 7,669,866 строк
      Обработано 4,000,000 / 7,669,866 строк
      Обработано 5,000,000 / 7,669,866 строк
      Обработано 6,000,000 / 7,669,866 строк
      Обработано 7,000,000 / 7,669,866 строк
    Submission сохранен: src\submission\custom_catboost_optuna_submission.csv
    Размер: (7669866, 2)
    Диапазон предсказаний: [0.000004, 0.969301]
    Среднее предсказание: 0.122738



## 7. Результаты на Kaggle

Все разработанные модели были отправлены на платформу Kaggle для оценки на тестовой выборке. Ниже представлены результаты всех submission, они немного лучше чем результаты в ноутбуке, так как были попытки обучения на больших обьемах данных

![kaggle](readme_files/kaggle.png)

- Наилучший результат показал catboost с optuna `0.89518`
- LightGBM c optuna `0.88089`
- Бейзлайн на LightAutoML `0.87945`

### 7.2. Анализ позиции в рейтинге

Анализируем позицию лучших результатов относительно других участников соревнования


```python
leaderboard = pd.read_csv('../playground-series-s4e7-publicleaderboard-2025-12-16T20_03_46.csv')
leaderboard_sorted = leaderboard.sort_values('Score', ascending=False).reset_index(drop=True)
total = len(leaderboard)

results = {
    'OptunaCatboost': 0.89541,
    'SimpleСatboost': 0.88993,
    'AdvancedCatboost': 0.88929,
    'LightGBM': 0.88141,
    'LightAutoML': 0.87958
}

analysis = []
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    rank = leaderboard_sorted[leaderboard_sorted['Score'] <= score].index[0] + 1
    top_pct = len(leaderboard_sorted[leaderboard_sorted['Score'] > score]) / total * 100
    analysis.append({
        'Submission': name, 
        'Public Score': score, 
        'Rank': rank, 
        'Top %': round(top_pct, 2)
    })

results_df = pd.DataFrame(analysis)

best_score = leaderboard['Score'].max()
worst_score = leaderboard['Score'].min()
mean_score = leaderboard['Score'].mean()
median_score = leaderboard['Score'].median()

print(f"Всего участников: {total}\n")
print("Статистика лидерборда:")
print(f"Лучший результат: {best_score:.6f}")
print(f"Медианный результат: {median_score:.6f}")
print(f"Средний результат: {mean_score:.6f}")
print("\nПозиции в рейтинге:\n")
print(results_df.to_string(index=False))
```

    Всего участников: 2236
    
    Статистика лидерборда:
    Лучший результат: 0.897930
    Медианный результат: 0.876265
    Средний результат: 0.806191
    
    Позиции в рейтинге:
    
          Submission  Public Score  Rank  Top %
      OptunaCatboost       0.89541   339  15.12
      SimpleСatboost       0.88993   507  22.63
    AdvancedCatboost       0.88929   520  23.21
            LightGBM       0.88141   719  32.11
         LightAutoML       0.87958   925  41.32
    

## 8. Выводы и заключение

### Основные результаты:

1. **EDA анализ** показал дисбаланс классов и важность некоторых признаков для предсказания целевой переменной

2. **LightAutoML baseline** показал хорошие результаты с двумя разными конфигурациями. Это подтверждает что автоматизированные подходы работают хорошо

3. **Собственное решение** с CatBoost дало значительное улучшение результатов по сравнению с baseline. Простой подход с минимальной предобработкой оказался самым эффективным

4. **Feature engineering** не дал значительного улучшения качества модели. Это значит что базовых признаков достаточно для этой задачи

5. **Оптимизация гиперпараметров** через Optuna помогла дополнительно улучшить качество модели

### Ключевые выводы:

* CatBoost показал лучшие результаты потому что хорошо работает с категориальными признаками
* Минимальная предобработка данных оказалась эффективнее чем сложные преобразования
* Оптимизация гиперпараметров важна для достижения максимального качества
* Простые подходы могут быть эффективнее чем сложные feature engineering техники

### Дополнительные исследования:

Во время исследования были проведены дополнительные эксперименты которые не показаны в этом блокноте:

* Разные варианты ансамблей моделей: stacking, blending
* Другие методы очистки данных и обработки аномалий
* Эксперименты с другими алгоритмами градиентного бустинга

Эти исследования подтвердили что простой CatBoost подход с оптимизацией гиперпараметров это лучшее решение для этой задачи.

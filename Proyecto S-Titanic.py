
# coding: utf-8

# # Analisis de supervivencia en el RMS Titanic
# El RMS Titanicnota 1 fue un transatlántico británico, el mayor barco del mundo al finalizar su construcción, que se hundió en la noche del 14 a la madrugada del 15 de abril de 1912 durante su viaje inaugural desde Southampton a Nueva York. En el hundimiento del Titanic murieron 1514 personas de las 2223 que iban a bordo, lo que convierte a esta tragedia en uno de los mayores naufragios de la historia ocurridos en tiempo de paz. Construido entre 1909 y 1912 en el astillero Harland and Wolff de Belfast, el Titanic era el segundo de los tres transatlánticos que formaban la clase Olympic, propiedad de la naviera White Star Line, junto al RMS Olympic y, posteriormente, el HMHS Britannic.
# 
# A continuación, se utilizara un set de datos el cual contiene un listado de los pasajeros que estuvieron abordo del Titanic para anlizar la supervivenvia de los pasajeros segun ciertas caracteristicas (Sexo, edad, cabina, embarcación, entre otras).
# 
# En este Notebook, utilizaremos ademas, la libreria Scikit-Learn para el procesamiento de datos, y la predicción de nuevos datos sin etiquetar (Determinar la probabilidad de supervivencia) los cuales se encuantran en el conjunto de datos de prueba.

# In[33]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


data_train = pd.read_csv('train.csv') # Datos de entrenamiento
data_test = pd.read_csv('test.csv') # Datos de Prueba


# In[35]:


data_train.sample(3)


# ## Visualización de datos

# In[36]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)


# In[37]:


sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
                 palette={"male": "blue", "female": "pink"},
                  markers=["*", "o"], linestyles=["-", "--"])


# ## Transformando caracteristicas

# In[38]:


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()


# In[39]:


sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train)


# In[40]:


sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train)


# In[41]:


sns.barplot(x="Fare", y="Survived", hue="Sex",data=data_train)


# ## Codificación Final

# La última parte de la fase de preprocesamiento es normalizar las etiquetas. El LabelEncoder en Scikit-learn convertirá cada valor de cadena único en un número, lo que hace que los datos sean más flexibles para varios algoritmos.
# 
# El resultado es una tabla de números que da miedo a los humanos, pero es hermosa para las máquinas.

# In[42]:


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
data_train.head()


# ## División de los datos de entrenamiento
# Ahora es tiempo de que las maquinas aprendan.
# 
# Primero, separe las características (X) de las etiquetas (y).
# 
# X_all: todas las características menos el valor que queremos predecir (Sobrevivido).
# 
# y_all: solo el valor que queremos predecir
# 
# Segundo, use Scikit-learn para mezclar aleatoriamente estos datos en cuatro variables. En este caso, estoy entrenando el 80% de los datos, luego probando contra el otro 20%.
# 
# Más tarde, estos datos se reorganizarán en un patrón de KFold para validar la efectividad de un algoritmo entrenado.

# In[43]:


from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# ## Entrenando al Algoritmo
# Ahora es el momento de descubrir qué algoritmo ofrecerá el mejor modelo. Voy con el RandomForestClassifier, pero puedes soltar cualquier otro clasificador aquí, como Support Vector Machines o Naive Bayes.

# In[44]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)


# In[29]:


predicton = clf.predict(X_test)
accuracy_score(y_test ,predicton)


# ## Predecir los datos de prueba reales

# In[32]:


ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
# output.to_csv('titanic-predictions.csv', index = False)
output.head(10)


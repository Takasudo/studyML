import pandas as pd
print("")

# 4.2.1
df = pd.DataFrame([
        ['green','M',10.1,'class1'],
        ['red','L',13.5,'class2'],
        ['blue','XL',15.3,'class1']
        ])

df.columns = ['color','size','price','classlabel']
print(df)
print("")

# 4.2.2 (size)

size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print(size_mapping.items())
print(df)
print("")

# 4.2.3 (classlabel)

import numpy as np
print(np.unique(df['classlabel']))
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
print("")

inv_class_mapping = {v:k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
print("")

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))
print("")

# 4.2.4 one-hot

X = df[['color','size','price']].values
print(X)
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features=[0])
tmp = one.fit_transform(X).toarray()
print(tmp)
print("")

pdDummy = pd.get_dummies(df[['price','color','size']])
print(pdDummy)

pdDummy_drop = pd.get_dummies(df[['price','color','size']], drop_first=True)
print(pdDummy_drop)

print("")

import pandas as pd
print("")

from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0'''

df = pd.read_csv(StringIO(csv_data))

print(df)
print("")
print(df.isnull().sum())
print("")
print(df.dropna())
print("")
print(df.dropna(axis=1))

from sklearn.impute import SimpleImputer

imr = SimpleImputer(strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print("")
print(imputed_data)



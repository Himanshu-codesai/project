import pandas as pd
import numpy as np


books = pd.read_csv('BX-Books.csv',encoding='ISO-8859-1',low_memory=False)  #To encode unicodedecodeerror(encoding='ISO-8859-1',low_memory=False)
# print(books.info)
# print(books.head())

# print(books.isnull().sum())
books = books.dropna()

bookratings = pd.read_csv('BX-Book-Ratings.csv',encoding='ISO-8859-1',low_memory=False)
# print(bookratings.info)

users = pd.read_csv('BX-Users.csv',encoding='ISO-8859-1',low_memory=False)
# print(users.head())

df=bookratings.merge(books, on="isbn")
# print(df.head())
# print(df['user_id'])

df['user_id'] = df['user_id'].astype('int')
# print(df['user_id'])

# df.groupby('user_id')['book_title'].agg('count').sort_values()
# print(df.groupby('user_id')['book_title'].agg('count').sort_values())
# print(df.groupby('isbn')['book_title'].agg('count').sort_values())
# print(df.groupby('isbn')['user_id'].agg('count').sort_values())

isbn = df['isbn'].tolist()
user_id= df['user_id'].tolist()
# print(isbn)

df1 = df.reindex(columns=["isbn", "user_id", "book_title", "book_author", "year_of_publication", "publisher", "rating"])
# print(df1.head(5))
# print(df1.columns)
# print(df1.head(100))

x=pd.DataFrame(df1['user_id'])
y=pd.DataFrame(df1['rating'])
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)
y_pred =regressor.predict(x_test)
print(y_pred)


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

r2=r2_score(y_test,y_pred)
print(r2)

print('MAE',mean_absolute_error(y_test,y_pred))

print('MSE',mean_squared_error(y_test,y_pred))

from math import sqrt as sqrt
print('RMSE',sqrt(mean_squared_error(y_test,y_pred)))


import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


def drop_columns(df, work_columns):
    return df.drop(work_columns, axis=1)

def remove_missing(df):
    return df.dropna()

def fill_missing_mean(df, work_columns):
    for col in work_columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def fill_missing_min(df, work_columns):
    for col in work_columns:
        df[col] = df[col].fillna(df[col].min())
    return df

def fill_missing_max(df, work_columns):
    for col in work_columns:
        df[col] = df[col].fillna(df[col].max())
    return df

def fill_missing_zero(df, work_columns):
    for col in work_columns:
        df[col] = df[col].fillna(0)
    return df

def fill_missing_mode(df):
    df = df.fillna(df.mode().iloc[0])
    return df

def fill_missing_regression(df, numeric):
    for col in numeric:
        df_num = df[numeric]
        test_data = df_num[df_num[col].isnull()]
        df_num = df_num.dropna()
        x_train = df_num.drop(col,axis=1)
        y_train = df_num[col]
        lr = LinearRegression()
        lr.fit(x_train,y_train)
        test_col = []
        for i in numeric:
            if(i != col):
                test_col.append(i)
        x_test = test_data[test_col]
        x_test = fill_missing_mean(x_test,test_col)
        y_pred = lr.predict(x_test)
        test_data[col] = y_pred
        for i in test_data.index.values:
            df.at[i,col] = test_data.loc[i][col]
    return df

def standardize(df, work_columns):
    standard_scaler = preprocessing.StandardScaler()
    for col in work_columns:
        values = df[col].values
        df_scaled = standard_scaler.fit_transform(values.reshape(-1, 1)) 
        df_scaled = pd.DataFrame(df_scaled)
        df[col] = df_scaled
    return df

def normalize(df, work_columns):
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in work_columns:
        values = df[col].values
        df_scaled = min_max_scaler.fit_transform(values.reshape(-1, 1)) 
        df_scaled = pd.DataFrame(df_scaled)
        df[col] = df_scaled
    return df

def remove_outliers_lof(df,work_columns):
    df_temp = df
    df_temp = df_temp.loc[:, work_columns]
    clf = LocalOutlierFactor(n_neighbors=2)
    clf.fit(df_temp)
    y_pred_outliers = clf.fit_predict(df_temp)
    df_temp['outlier'] = y_pred_outliers

    df_temp = df_temp.loc[df_temp['outlier'] == 1]
    df_temp.drop('outlier', axis=1, inplace=True)
    df_temp = df_temp.reset_index(drop=True)
    df = df[df.index.isin(df_temp.index)]
    return df

def encode_categorical(df,work_columns):
    encoder = LabelEncoder()
    for col in work_columns:
        df[col] = encoder.fit_transform(df[col])
    return df
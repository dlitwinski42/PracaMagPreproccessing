import pandas as pd
import random

def prepare_to_file(df,work_columns,filename,count):
    path=r'E:\Magisterka\PracaMagPreproccessing\Datasets\Prepared'
    for i in range(count):
        df_copy = df.copy()
        df_random_rows = df.sample(frac=0.1)
        random_indexes = df_random_rows.index.tolist()
        for j in random_indexes:
            chosen_column = random.choice(work_columns)
            df_copy.at[j, chosen_column] = pd.NA
        df_copy.to_csv(path + '\\' + filename +'_'+str(i+1)+'.csv', index=False)

def prepare_and_return(df,work_columns):
    df_copy = df.copy()
    df_random_rows = df.sample(frac=0.1)
    random_indexes = df_random_rows.index.tolist()
    for i in random_indexes:
        chosen_column = random.choice(work_columns)
        df_copy.at[i, chosen_column] = pd.NA
    return df_copy
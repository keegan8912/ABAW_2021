import pandas as pd
import pickle
import os
from tqdm import tqdm
import random

with open("/var/storage/cube-data/others/df_collection_clean3.pickle", "rb") as input_file:
    df2 = pickle.load(input_file)

path_ ='/var/storage/cube-data/others/Code/images/cls'

df_3 = df2[df2.labels == 3]
df2 = pd.concat([df2, df_3]) #taken twice

df_2 = df2[df2.labels == 2]
df2 = pd.concat([df2, df_2.sample(frac=0.75).reset_index(drop=True)]) #

df_1 = df2[df2.labels == 1]
df2 = pd.concat([df2, df_1.sample(frac=0.75).reset_index(drop=True)]) #

classes = [2,3]

samples = {
    1: 22808//2,
    2: 21826//2,
    3: 21694//2
}


for i in tqdm(classes):
    path_i = path_+str(i)
    images = os.listdir(path_i)
    p = [os.path.join(path_i, img) for img in images]
    p = random.sample(p, samples[i])

    df3 = pd.DataFrame()
    df3 = pd.DataFrame(p, columns=['Files'])
    df3['labels'] = i
    df2 = pd.concat([df2, df3])
    df2 = df2.sample(frac=1).reset_index(drop=True)

cls_1_df = df2[df2.labels == 0].iloc[::5, :]
cls_4_df = df2[df2.labels == 4].iloc[::4, :]
cls_5_df = df2[df2.labels == 5].iloc[::3, :]

df_reduced = df2[(df2.labels != 0) & (df2.labels != 4) & (df2.labels != 5)]
df_reduced = pd.concat([df_reduced, cls_1_df, cls_4_df, cls_5_df])
df_reduced = df_reduced.sample(frac=1).reset_index(drop=True)
with open('/var/storage/cube-data/others/extnd_23.pickle', 'wb') as op:
    pickle.dump(df_reduced, op)



import pandas as pd
import numpy as np

#load data
df = pd.read_csv('datset/training.csv')

print('Number of training articles: {:,}\n'.format(df.shape[0]))

# Get the lists of sentences and their labels.
sentences = df['QUESTION'].values
labels = df['ANSWER'].values

sentences = sentences
labels = labels

ids=[]
masks=[]
encode=[]
cnt=0
combine = ''
combine_test = ''

for i in range(len(sentences)):
    cnt+=1
    print('Encoding the tokens..',cnt)
    sent = sentences[i].replace('[BLANK]',labels[i],1)
    if(cnt<80000):
        combine = combine+"\n\n"+sent
    else:
        combine_test = combine_test+'\n\n'+sent

text_file = open("training.raw", "w")
n = text_file.write(combine)
text_file.close()

text_file = open("testing.raw", "w")
n = text_file.write(combine_test)
text_file.close()
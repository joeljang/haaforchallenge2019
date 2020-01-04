import torch
import tensorflow
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import os
import wget
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# We'll borrow the `pad_sequences` utility function to do this.
from tensorflow.keras.preprocessing.sequence import pad_sequences

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#load data
df = pd.read_csv('datset/training.csv')

print('Number of training articles: {:,}\n'.format(df.shape[0]))

# Get the lists of sentences and their labels.
sentences = df['QUESTION'].values
labels = df['ANSWER'].values

sentences = sentences[0:10000]
labels = labels[0:10000]

finetunedir = './outputbert/'
# Load the BERT tokenizer.
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# Load a trained model and vocabulary that you have fine-tuned
model = BertForMaskedLM.from_pretrained(finetunedir)
tokenizer = BertTokenizer.from_pretrained(finetunedir)

# Copy the model to the GPU.
model.to(device)


ids=[]
masks=[]
encode=[]
cnt=0

for sent in sentences:
    print('Encoding the tokens..')
    cnt+=1
    print(cnt)
    sent1= sent.replace('[BLANK]','[MASK]',1)
    encoded = tokenizer.encode(sent1,add_special_tokens=True,max_length=128, return_tensors='pt')
    encoded_n = encoded.numpy()
    encode.append(encoded_n)
    ids.append(encoded)
    segments_ids = [0] * len(encoded)
    segments_tensors = torch.tensor([segments_ids])
    masks.append(segments_tensors)

# Tell pytorch to run this model on the GPU.
model.cuda()

model.eval()

correct=0
cnt=0
for i in range(len(ids)):
    cnt+=1
    print(correct, cnt)
    # Predict all tokens
    b_ids = ids[i].to(device)
    b_masks = masks[i].to(device)
    model.zero_grad()  
    with torch.no_grad():
        predictions = model(b_ids, b_masks)
    
    index=0
    for e in encode[i][0]:
        if(e==103):
            masked_index = index
        else:
            index+=1

    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token,labels[i])
    if(predicted_token == labels[i]):
        correct+=1

print('Number of correct answers with fine-tuning out of 90,000: ',correct)
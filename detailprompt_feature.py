from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from tsai.all import *
import pandas as pd
import numpy as np
import os 

def split_string(my_string):
    words = my_string.split()
    substrings = []
    current_substring = ""
    for word in words:
        if len(current_substring) + len(word) <= 512:
            current_substring += word + " "
        else:
            substrings.append(current_substring.strip())
            current_substring = word + " "
    substrings.append(current_substring.strip())
    return substrings

def find_max_values(my_list):
    max_values = []
    for col in range(len(my_list[0])):
        max_value = my_list[0][col]
        for row in range(1, len(my_list)):
            if my_list[row][col] > max_value:
                max_value = my_list[row][col]
        max_values.append(max_value)
    return max_values

def number_to_ordinal(num):
    suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
    if 10 < num % 100 < 20:
        suffix = 'th'
    else:
        suffix = suffixes.get(num % 10, 'th')
    return str(num) + suffix

target_list = get_UCR_univariate_list() 
# target_list = ['MixedShapesRegularTrain']
model_path = './model/models--bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained('./model/models--roberta-base')
# model = AutoModel.from_pretrained("./model/models--roberta-base")

for t in target_list:
    print(t, "start")
    if t not in os.listdir("./dataset"):
        os.mkdir("./dataset/" + t)
    # if 'train.csv' in os.listdir("./dataset/"  + t) and 'test.csv' in os.listdir("./dataset/"  + t):
    #     continue
    X_train, y_train, X_test, y_test  = get_UCR_data(t, return_split=True)
    
    features = []
    pool_feature = []
    text = "The length of time series is tslen. The original time series is splited into subtsnum sub series, whose length is subslen. The specific value of the subsindex sub series are [] in order."
    for sample in X_train:
        tslen = len(sample[0])
        tstext = ""
        
        for s in sample[0]:
            # print(s)
            tstext += str(s) + " "
        tstext = tstext.rstrip()
        subtsnum = len(split_string(tstext))
        for isp in split_string(tstext):
            subslen = len(isp.split(" "))
            subsindex = number_to_ordinal(split_string(tstext).index(isp) + 1)
            subsdetail = isp
            subts_text = text.replace("tslen", str(tslen)).replace("subtsnum", str(subtsnum)).replace("subslen", str(subslen)).replace("subsindex", str(subsindex)).replace("[]", str(subsdetail))
            # print(subts_text)
       
            encoded_input = tokenizer(subts_text, return_tensors='pt')
            output = model(**encoded_input)
            output = np.array(output.pooler_output.detach().numpy())
            pool_feature.append(output[0])
        max_pool = find_max_values(pool_feature)
        features.append(max_pool)
    df = pd.DataFrame(features)
    df.to_csv("./dataset/" + t + "/ddp_bert_train.csv", index=False)

    index = 0
    features = []
    pool_feature = []
    text = "The length of time series is tslen. The original time series is splited into subtsnum sub series, whose length is subslen. The specific value of the subsindex sub series are [] in order."
    for sample in X_test:
        print(index, len(X_test))
        tslen = len(sample[0])
        tstext = ""
        
        for s in sample[0]:
            # print(s)
            tstext += str(s) + " "
        tstext = tstext.rstrip()
        subtsnum = len(split_string(tstext))
        for isp in split_string(tstext):
            subslen = len(isp.split(" "))
            subsindex = number_to_ordinal(split_string(tstext).index(isp) + 1)
            subsdetail = isp
            subts_text = text.replace("tslen", str(tslen)).replace("subtsnum", str(subtsnum)).replace("subslen", str(subslen)).replace("subsindex", str(subsindex)).replace("[]", str(subsdetail))
            # print(subts_text)
       
            encoded_input = tokenizer(subts_text, return_tensors='pt')
            output = model(**encoded_input)
            output = np.array(output.pooler_output.detach().numpy())
            pool_feature.append(output[0])
        max_pool = find_max_values(pool_feature)
        features.append(max_pool)
        index += 1
    df = pd.DataFrame(features)
    df.to_csv("./dataset/" + t + "/ddp_bert_test.csv", index=False)

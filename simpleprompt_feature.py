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

target_list = get_UCR_univariate_list()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

for t in target_list:
    print(t, "start")
    if t not in os.listdir("./dataset"):
        os.mkdir("./dataset/" + t)
    # if 'train.csv' in os.listdir("./dataset/"  + t) and 'test.csv' in os.listdir("./dataset/"  + t):
    #     continue
    X_train, y_train, X_test, y_test  = get_UCR_data(t, return_split=True)
    
    features = []
    for sample in X_train:
        text = ""
        for s in sample[0]:
            text += str(s) + " "
        text = text.rstrip()
        # print(len(split_string(text)))
        pool_feature = []
        for i in split_string(text):
            encoded_input = tokenizer(i, return_tensors='pt')
            output = model(**encoded_input)
            output = np.array(output.pooler_output.detach().numpy())
            pool_feature.append(output[0])
        max_pool = find_max_values(pool_feature)
        features.append(max_pool)
    df = pd.DataFrame(features)
    df.to_csv("./dataset/" + t + "/sdp_bert_train.csv", index=False)

    features = []
    for sample in X_test:
        text = ""
        for s in sample[0]:
            text += str(s) + " "
        text = text.rstrip()
        # print(len(split_string(text)))
        pool_feature = []
        for i in split_string(text):
                encoded_input = tokenizer(i, return_tensors='pt')
                output = model(**encoded_input)
                output = np.array(output.pooler_output.detach().numpy())
                pool_feature.append(output[0])
        max_pool = find_max_values(pool_feature)
        features.append(max_pool)
    df = pd.DataFrame(features)
    df.to_csv("./dataset/" + t + "/sdp_bert_test.csv", index=False)
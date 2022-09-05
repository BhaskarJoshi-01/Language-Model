import torch
from torch.utils.data import Dataset
import gensim.downloader
import numpy as np
train_path='./dataset/train.txt'
test_path='./dataset/test.txt'
val_path='./dataset/val.txt'

def load_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data

train_data = load_data(train_path)
test_data = load_data(test_path)
val_data = load_data(val_path)

# w2v_vectors = gensim.downloader.load('word2vec-google-news-300')
# print(dir(w2v_vectors))
# print(w2v_vectors.get_vector("hello"))
# print(len(w2v_vectors.get_vector("hello")))
# print(w2v_vectors.get_vector("ljkasdfalkj"))

class TextDataset(Dataset):
    def __init__(self, data,dict):
        self.data = data
        self.new_data = []
        for i in range(len(self.data)):
            self.data[i] = self.data[i].strip("\n")
            self.data[i] = self.data[i].split(" ")
            token= [ "#pjrocks","#pkgiri","#tswift","#itswhatitis"]
            token.extend(self.data[i])
            for k in range(5,len(token)+1):
                self.new_data.append(token[k-5:k])
        t=[]
        if dict==None:
            self.dict = {}
            for i in range(len(self.new_data)):
                t.extend(self.new_data[i])
            self.vocab = list(set(t))
            for i in range(len(self.vocab)):
                self.dict[self.vocab[i]]=i
        else:
            self.dict=dict

    # def get_vector(self,word):
    #     ret_var=[]
    #     try:
    #         ret_var= w2v_vectors.get_vector(word)
    #     except KeyError as e:
    #         ret_var= np.zeros(300)
    #     ret_var.setflags(write=True)
    #     return torch.from_numpy(ret_var)
    
    def __len__(self):
        return len(self.new_data)

    def __getitem__(self,idx):
        token = self.new_data[idx]
        print(token)
        if len(token)!=5:
            print("error")
            exit(2)
        x = []
        for i in range(4):
            # x.append(self.get_vector(token[i]))
            x.append(torch.zeros(300))
        x = torch.cat(x)
        if token[4] in self.dict:
            y = self.dict[token[4]]
        else:
            y = self.dict["#pjrocks"]
        if len(x)!=1200:
            print("error")
            exit(2)
        return x
        


train_dataset = TextDataset(train_data,None)
test_dataset = TextDataset(test_data,train_dataset.dict)
val_dataset = TextDataset(val_data,train_dataset.dict)
dict_len=len(train_dataset.dict)
# print(len(train_dataset[0]))
# print(len(train_dataset[1]))
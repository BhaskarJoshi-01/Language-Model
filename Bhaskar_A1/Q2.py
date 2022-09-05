from cProfile import run
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
# from load_data import dict_len

from rnn import MyRnn
from gru import MyGRU



import gensim.downloader
import numpy as np
train_path='./dataset/train.txt'
test_path='./dataset/test.txt'
val_path='./dataset/val.txt'


train = False
var=1 # 0 for RNN, 1 for GRU
if var==0:
    save_name="model_rnn"
else:
    save_name="model_gru"

def load_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data

train_data = load_data(train_path)
test_data = load_data(test_path)
val_data = load_data(val_path)

w2v_vectors = gensim.downloader.load('word2vec-google-news-300')
# print(dir(w2v_vectors))
# print(w2v_vectors.get_vector("hello"))
# print(len(w2v_vectors.get_vector("hello")))
# print(w2v_vectors.get_vector("ljkasdfalkj"))

class TextDataset(Dataset):
    def __init__(self, data,dict):
        self.data = data
        for i in range(len(self.data)):
            self.data[i] = self.data[i].strip("\n")
            self.data[i] = self.data[i].split(" ")
        
        mx_len=max([len(i) for i in self.data])
        for i in range(len(self.data)):
            self.data[i].extend(["#pjrocks"]*(mx_len-len(self.data[i])))
            self.data[i].insert(0,"#341")
        
            # token= [ "#pjrocks","#pkgiri","#tswift","#itswhatitis"]
            # token.extend(self.data[i])
            # for k in range(5,len(token)+1):
            #     self.new_data.append(token[k-5:k])

        t=[]
        if dict==None:
            self.dict = {}
            for i in range(len(self.data)):
                t.extend(self.data[i])
            self.vocab = list(set(t))
            for i in range(len(self.vocab)):
                self.dict[self.vocab[i]]=i
            self.save_dict()
        else:
            self.dict=dict

    def get_vector(self,word):
        ret_var=[]
        try:
            ret_var= w2v_vectors.get_vector(word)
            ret_var.setflags(write=True)
            ret_var = torch.from_numpy(ret_var)
        except KeyError as e:
            ret_var= torch.zeros(300)
        return ret_var
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        token = self.data[idx]
        x = []
        target=[]
        for i in range(len(token)-1):
            x.append(self.get_vector(token[i]))
        x = torch.stack(x)
        for i in range(1,len(token)):
            if token[i] in self.dict:
                y = self.dict[token[i]]
            else:
                y = self.dict["#pjrocks"]
            target.append(y)
        return x, torch.tensor(target)
        
    
    def get_unpadded(self, idx):
        token = self.data[idx]
        sentence=[]
        for itr in token:
            if itr != "#pjrocks":
                sentence.append(itr)
        
        x = []
        target=[]
        for i in range(len(sentence)-1):
            x.append(self.get_vector(sentence[i]))
        x = torch.stack(x)
        for i in range(1,len(sentence)):
            if sentence[i] in self.dict:
                y = self.dict[sentence[i]]
            else:
                y = self.dict["#pjrocks"]
            target.append(y)
        return " ".join(sentence[1:]), x, torch.tensor(target)
        

    def save_dict(self):
        with open('dict'+str(var)+'.json', 'w') as fp:
            json.dump(self.dict, fp)

            # calc_val+=self.__getitem__(k-5)[1]*self.__getitem__(k-4)[1]*self.__getitem__(k-3)[1]*self.__getitem__(k-2)[1]*self.__getitem__(k-1)[1]


vocabulary = None
if train == False:
    with open('dict'+str(var)+'.json') as f:
        vocabulary = json.load(f)

train_dataset = TextDataset(train_data, vocabulary)
test_dataset = TextDataset(test_data,train_dataset.dict)
val_dataset = TextDataset(val_data,train_dataset.dict)
dict_len=len(train_dataset.dict)
# print(len(train_dataset[0]))
# print(len(train_dataset[1]))







#defining the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=20)
val_dataloader = DataLoader(val_dataset, batch_size=64,num_workers=20)
test_dataloader = DataLoader(test_dataset, batch_size=64,num_workers=20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model


if train:   
    if var==0:
        net=MyRnn(300,300,dict_len).to(device)
    else:
        net=MyGRU(300,300,dict_len).to(device)
else:
    if var==0:  
        net=MyRnn(300,300,dict_len).to(device)
    else:
        net=MyGRU(300,300,dict_len).to(device)
    net.load_state_dict(torch.load(save_name+".pt"))
    # net = MyRnn(300,300,dict_len)
    # net.load_state_dict(torch.load("model2.pth"))
    net = net.to(device)
    net.eval()
# Define the loss function
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Test the model
def accuracy(dataloader):
    net.eval()
    running_loss = 0
    correct_pred = 0
    total_ = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data # B * 1200, B*1

        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs=torch.transpose(inputs,0,1)
        labels=torch.transpose(labels,0,1)
        hidden_state=torch.zeros(labels.size()[1],300).to(device)
        

        for j in range(len(labels)):
            outputs, hidden_state = net(inputs[j],hidden_state) 
            loss=criterion(outputs,labels[j])/len(labels)
            for k in range(outputs.size()[0]):
                if torch.argmax(outputs[k])==labels[j][k]:
                    correct_pred+=1
                total_+=1
            running_loss+=loss.item()

    running_loss /= len(dataloader)
    net.train()
    print("Loss: ", running_loss, "Accuracy", correct_pred / total_, " ",save_name)


if train:
    # Train the model
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs=torch.transpose(inputs,0,1)
            labels=torch.transpose(labels,0,1)
            total_loss=0
            hidden_state=torch.zeros(labels.size()[1],300).to(device)
            
            for j in range(len(labels)):
                outputs, hidden_state =net(inputs[j],hidden_state)
                loss=criterion(outputs,labels[j])
                total_loss+=loss

            # zero the parameter gradients
            optimizer.zero_grad()
            total_loss/=len(labels)
            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += total_loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        accuracy(val_dataloader)
    print('Finished Training')
    accuracy(test_dataloader)
    torch.save(net.state_dict(), save_name+'.pt')



def ppl(pth, dataset):
    net.eval()
    file = open(pth, "w+")
    totalppl = 0
    total_Correct =0
    total_count = 0
    
    for i in range(len(dataset.data)):
        sent, inputs, labels = dataset.get_unpadded(i)
        inputs = inputs.to(device) # B(# of ngrams for your sentence) * 1200
        labels = labels.to(device) # B * 1

        calc_ppl = 1
        hidden_state=torch.zeros(300).to(device)
        
        for j in range(len(labels)):
            outputs,hidden_state = net(inputs[j],hidden_state)   # B * V
            calc_ppl*= 1  / (torch.exp(outputs[labels[j]]) ** (1/len(labels)))
            if torch.argmax(outputs) == labels[j]:
                total_Correct += 1
            total_count += 1

        

        calc_ppl = calc_ppl.data.item()
        totalppl += calc_ppl
            

        file.write(sent + "\t" + str(calc_ppl) + "\n")

    file.write(str(totalppl / len(dataset.data)) + "\n")
    file.close()
    print("Accuracy: ", total_Correct / total_count)
    
#can change name accordingly to rnn or gru
ppl("2019111002-LM2-train-perplexity_"+save_name+".txt", train_dataset)
ppl("2019111002-LM2-test-perplexity_"+save_name+".txt", test_dataset)
ppl("2019111002-LM2-val-perplexity_"+save_name+".txt", val_dataset)

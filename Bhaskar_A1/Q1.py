import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
# from load_data import dict_len






import gensim.downloader
import numpy as np
train_path='./dataset/train.txt'
test_path='./dataset/test.txt'
val_path='./dataset/val.txt'


train = True


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
        return len(self.new_data)

    def __getitem__(self,idx):
        token = self.new_data[idx]
        x = []
        for i in range(4):
            x.append(self.get_vector(token[i]))
        x = torch.cat(x)
        if token[4] in self.dict:
            y = self.dict[token[4]]
        else:
            y = self.dict["#pjrocks"]
        return x, y
        
    def get_sentence_ngrams(self, idx):
        sentence=self.data[idx]
        token= [ "#pjrocks","#pkgiri","#tswift","#itswhatitis"]
        token.extend(sentence)
        calc_val=0
        x=[]
        labels = []
        for k in range(5,len(token)+1):
            ngram = token[k-5:k]
            emb = []
            for i in range(4):
                emb.append(self.get_vector(ngram[i]))
            emb = torch.cat(emb)
            if ngram[4] in self.dict:
                y = self.dict[ngram[4]]
            else:
                y = self.dict["#pjrocks"]
            labels.append(y)
            x.append(emb)
        return " ".join(sentence), x, labels
    
    def save_dict(self):
        with open('dict.json', 'w') as fp:
            json.dump(self.dict, fp)

            # calc_val+=self.__getitem__(k-5)[1]*self.__getitem__(k-4)[1]*self.__getitem__(k-3)[1]*self.__getitem__(k-2)[1]*self.__getitem__(k-1)[1]


vocabulary = None
if train == False:
    with open('dict.json') as f:
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
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(4*300,300)
        self.linear2 = nn.Linear(300,300)
        self.linear3 = nn.Linear(300,dict_len)
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x=self.linear1(x)
        x=self.tanh(x)
        x=self.linear2(x)
        x=self.tanh(x)
        x=self.linear3(x)
        x=self.logsoftmax(x)
        return x

if train:   
    net=Model().to(device)
else:
    net = Model()
    net.load_state_dict(torch.load("model.pth"))
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
        optimizer.zero_grad()

        outputs = net(inputs) # B * V
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        for j in range(len(labels)):
            out = outputs[j]
            lbl = labels[j]
            # if the place where out is max is at lbl the correct += 1
            if torch.argmax(out) == lbl:
                correct_pred += 1
            total_ += 1

            # mx=-1e5
            # idx=0
            # for k in range(len(outputs[j])):
            #     if(outputs[j][k]>mx):
            #         mx=outputs[j][k]
            #         idx=k
            # if idx == lbl:
            #     correct_pred+=1
            # total_ += 1
    running_loss /= len(dataloader)
    net.train()
    print("Loss:", running_loss, "Accuracy", correct_pred / total_)


if train:
    # Train the model
    for epoch in range(8):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        accuracy(val_dataloader)
    print('Finished Training')
    accuracy(test_dataloader)
    torch.save(net.state_dict(), './model.pth')



def ppl(pth, dataset):
    net.eval()
    file = open(pth, "w+")
    totalppl = 0
    for i in range(len(dataset.data)):
        sentence, inputs, labels = dataset.get_sentence_ngrams(i)
        inputs = torch.stack(inputs).to(device) # B(# of ngrams for your sentence) * 1200
        labels = torch.tensor(labels).to(device) # B * 1

        outputs = net(inputs)   # B * V
        # for all B calc P for the corresponding label and multiply
        calc_ppl = 1
        for j in range(len(labels)):
            calc_ppl*= 1  / (torch.exp(outputs[j, labels[j]]) ** (1/len(labels)))
        

        calc_ppl = calc_ppl.data.item()
        totalppl += calc_ppl
        file.write(sentence + "\t" + str(calc_ppl) + "\n")

    file.write(str(totalppl / len(dataset.data)) + "\n")
    file.close()

ppl("2019111002-LM1-train-perplexity.txt", train_dataset)
ppl("2019111002-LM1-test-perplexity.txt", test_dataset)
ppl("2019111002-LM1-val-perplexity.txt", val_dataset)

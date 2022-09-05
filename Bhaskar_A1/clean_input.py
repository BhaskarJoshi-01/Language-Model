#convert input text to sentences
import re
import random
random.seed(12)

def preprocess(sentences):
    sentences = sentences.lower()
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', sentences)        
    new_list=[]
    for i in range(len(sentences)):
        sentences[i] = re.sub(r'[^a-zA-Z\s]', ' ', sentences[i])
        sentences[i] = sentences[i].strip()
        sentences[i] = sentences[i].split()
        if sentences[i]:
            if sentences[i] != []:
                new_list.append(" ".join([ word for word in sentences[i] if word!='' ] ))

    return new_list

with open('input.txt') as f:
    text = f.read()
    sentence=preprocess(text)
    random.shuffle(sentence)
    train,test,val=sentence[:int(len(sentence)*0.7)],sentence[int(len(sentence)*0.7):int(len(sentence)*0.8)],sentence[int(len(sentence)*0.8):]
    with open('train.txt', 'w+') as df:
        for i in train:
            df.write(i)
            df.write('\n')
    with open('test.txt', 'w+') as df:
        for i in test:
            df.write(i)
            df.write('\n')
    with open('val.txt', 'w+') as df:
        for i in val:
            df.write(i)
            df.write('\n')



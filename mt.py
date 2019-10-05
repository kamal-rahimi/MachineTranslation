
""""
pip3 install xlrd
3 install wordcloud


"""


import pandas as pd
import re
import numpy as np
import argparse

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS 

from sklearn.model_selection import train_test_split

SPLIT_PATTERN_WITH_DILIMITER = r'([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s])\s*'
SPLIT_PATTERN_NO_DILIMITER   = r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]\s*'


def plot_word_cloud(word_list):
    words = ' '.join(word_list)
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white',
                    stopwords = None,
                    collocations = False,
                    regexp=None,
                    min_word_length=0,
                    min_font_size = 10).generate(words) 
                         
    #plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()


def plot_length_distribution(list, max_length=100, cdf=False):
    length_count = [0 for _ in range(max_length)]
    #item =[list]
    for item in list:
        item_length = len(item)
        if item_length < max_length:
            length_count[item_length] += 1

    length_count = np.array(length_count)
    length_freq = length_count/np.sum(length_count)
    if cdf:
        length_freq = np.cumsum(length_freq)
    plt.plot(length_freq)
    plt.show()
"""
data_set = pd.read_excel("assignment/MT_training_corpus.xlsx")

#print(data_set["CONDITION"].values)

QIDs = data_set["QID"].values

CONDITIONs = data_set["CONDITION"].values

OUTPUTs = data_set["OUTPUT"].values
cond = [condition.split(" ") for condition in CONDITIONs]
print(cond[1:10][:])
plot_length_distribution(cond, cdf=False)

print(len(QIDs), " ", len(CONDITIONs), " ", len(OUTPUTs) )

all_cond = " ".join(CONDITIONs)
#all_cond = all_cond.split(" ")



all_cond = re.split(r'([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s])\s*', all_cond)

plot_word_cloud(all_cond)


all_out = [str(out) for out in OUTPUTs]
all_out = " ".join(all_out)

all_out = re.split(r'([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s])\s*', all_out)

#print(all_out)




dictionary = {}

for word in all_cond:
    dictionary[word] = dictionary.get(word, 0) + 1

dictionary = dict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True))

for word in dictionary:
    if (dictionary[word]>0):
        print(word, ": ", dictionary[word])



"""


def read_data(data_path):
    data_set = pd.read_excel("assignment/MT_training_corpus.xlsx")
    return data_set
    #print(data_set["CONDITION"].values)

def preprocess_sample(qid, condition, output):
    
    qid = re.split(SPLIT_PATTERN_NO_DILIMITER, str(qid))
    condition = re.split(SPLIT_PATTERN_WITH_DILIMITER, str(condition))
    condition = [cond for cond in condition if cond != " " and cond != ""]
    output = re.split(SPLIT_PATTERN_WITH_DILIMITER, str(output))

    qid       = [x.lower() for x in qid]
    condition = [x.lower() for x in condition]
    output    = [x.lower() for x in output]

    for index, id in enumerate(qid):
        standard_qid = 'QID{}'.format(index)
        
        qid[index] = standard_qid

        for word_index in range(len(condition)):
            if condition[word_index] == id:
                condition[word_index] = standard_qid

        for word_index in range(len(output)):
            if output[word_index] == id:
                output[word_index] = standard_qid

    digit_num = 0
    for word in condition:
        if word.isdigit():
            standard_digit = 'DIGIT{}'.format(digit_num)
            digit_num += 1
            for word_index in range(len(condition)):
                if condition[word_index] == word:
                    condition[word_index] = standard_digit

            for word_index in range(len(output)):
                if output[word_index] == word:
                    output[word_index] = standard_digit

    return qid, condition, output

def prepare_data(data_set):
    QIDs = data_set["QID"].values
    CONDITIONs = data_set["CONDITION"].values
    OUTPUTs = data_set["OUTPUT"].values

    QIDs_process = []
    CONDITIONs_process = []
    OUTPUTs_process = []
    for qid, condition, output in zip(QIDs, CONDITIONs, OUTPUTs):
        q, c, o = preprocess_sample(qid, condition, output)
        QIDs_process.append(q)
        CONDITIONs_process.append(c)
        OUTPUTs_process.append(o)
    
    print(QIDs_process[:2],"\n")
    print(CONDITIONs_process[:2],"\n")
    print(CONDITIONs[:2],"\n")
    print(OUTPUTs_process[:2])
    all_word = [word for i in range(len(CONDITIONs_process)) for word in CONDITIONs_process[i]]
    plot_word_cloud(all_word)
    all_word = [word for i in range(len(OUTPUTs_process)) for word in OUTPUTs_process[i]]
    plot_word_cloud(all_word)



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path",   type=str, default="./assignment/MT_training_corpus.xlsx", help="Specify the train data path")
    args = vars(ap.parse_args())
    train_data_path = args["path"]

    data_set = read_data(train_data_path)
    prepare_data(data_set)
   # X, y = prepare_data(data_set)
    #X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.2, random_state=42)




if __name__ == '__main__':
    main()




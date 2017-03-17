
# coding: utf-8


import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(text1, text2):
    lex = []
    with open(text1,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lex += list(all_words)
    
    with open(text2,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lex += list(all_words)
    
    lex = [lemmatizer.lemmatize(i) for i in lex]
    w_counts = Counter(lex)
    lexicon = []
    
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            lexicon.append(w)
    
    print('Featureset length ', len(lexicon))    
    return lexicon

def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon:
                    index_val = lexicon.index(word)
                    features[index_val] += 1
            
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)
    features = np.array(features)
    
    testing_size = int(test_size * len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])
    
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('data/pos.txt', 'data/neg.txt')
    # to picke this data
   # with open('data/sentiment_featureset.pickle', 'wb') as f:
    #    pickle.dump([train_x, train_y, test_x, test_y], f)





import nltk
import os
import pickle
from tqdm import tqdm
import pymorphy2
from pymongo import MongoClient
morph = pymorphy2.MorphAnalyzer()
#client = MongoClient('localhost', 27017)
client = MongoClient('localhost', 27017)
global db
#db = client.nell
db = client.nellFull

text_dictionary = dict()
path = '../resources/textsFull'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def process_text_for_patterns(max_n):
    for file in tqdm(files):
        f = open(path + '/' + file).read()
        sentences = nltk.sent_tokenize(f)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for i in range(1, max_n + 1):
                ngrams = nltk.ngrams(words, i)
                for ngram in ngrams:
                    s = ''
                    for word in ngram:
                        s += word
                        s += ' '
                    s = s[:-1].lower()
                    try:
                        text_dictionary[s] += 1
                    except:
                        text_dictionary[s] = 1
    with open('ngrams_dictionary_for_patterns.pkl', 'wb') as f:
        pickle.dump(text_dictionary, f)


def process_text_for_instances():
    for file in tqdm(files):
        f = open(path + '/' + file).read()
        sentences = nltk.sent_tokenize(f)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                lexem = morph.normal_forms(word)[0].lower()
                try:
                    text_dictionary[lexem] += 1
                except:
                    text_dictionary[lexem] = 1
    with open('ngrams_dictionary_for_instances.pkl', 'wb') as f:
        pickle.dump(text_dictionary, f)


def load_dictionary(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def calculate_lexems_in_sentences(db):
    ngrams_dicrionary_for_instances = dict()
    sentences = db['sentences'].find()
    for sentence in sentences:
        for word in sentence['words']:
            try:
                ngrams_dicrionary_for_instances[word['lexem']] += 1
            except:
                ngrams_dicrionary_for_instances[word['lexem']] = 1

    with open('ngrams_dictionary_for_instances.pkl', 'wb') as f:
        pickle.dump(ngrams_dicrionary_for_instances, f)

# process_text_for_patterns(files, 3)
# process_text_for_instances(files)
# calculate_lexems_in_sentences(db)
#load_dictionary('ngrams_dictionary_for_instances.pkl')

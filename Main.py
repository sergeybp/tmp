import sys

sys.path.insert(0, '../src/')
import helper
import logging
import nltk
import os
from tqdm import tqdm
import pymorphy2
from pymongo import MongoClient
import text_preprocesser

text_dictionary = dict()

morph = pymorphy2.MorphAnalyzer()

# FIXME enter full path for current files on your computer
ontology_path = 'categories_animals_ru.xls'
patterns_pool_path = 'patterns.xlsx'
log_path = 'log/cpl.log'
# texts_path = '../resources/texts'
texts_path = 'justTexts'

files = [f for f in os.listdir(texts_path) if os.path.isfile(os.path.join(texts_path, f))]

# FIXME end of path section
ITERATIONS = 100
db = None


def process_text_for_patterns(max_n):
    for file in tqdm(files):
        f = open(texts_path + '/' + file).read()
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
        for key, value in text_dictionary.items():
            item = db['ngramsP'].find_one({'string': key})
            if item is None:
                nItem = dict()
                nItem['string'] = key
                nItem['count'] = value
                db['ngramsP'].insert(nItem)
            else:
                tmp = item['count']
                db['ngramsP'].update({'string': key}, {'$set': {'count': (tmp + 1)}})


def process_text_for_instances():
    for file in tqdm(files):
        f = open(texts_path + '/' + file).read()
        sentences = nltk.sent_tokenize(f)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                lexem = morph.normal_forms(word)[0].lower()
                try:
                    text_dictionary[lexem] += 1
                except:
                    text_dictionary[lexem] = 1
        for key, value in text_dictionary.items() :
            item = db['ngramsS'].find_one({'string' : key})
            if item is None :
                nItem = dict()
                nItem['string'] = key
                nItem['count'] = value
                db['ngramsS'].insert(nItem)
            else :
                tmp = item['count']
                db['ngramsS'].update({'string' : key}, {'$set': {'count' : (tmp + value)}})


def connect_to_database():
    # client = MongoClient('localhost', 27017)

    client = MongoClient('localhost', 27017)
    global db
    db = client.nellFull


def inizialize():
    # Read initial ontology and patterns
    logging.basicConfig(filename=log_path, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    helper.get_patterns_from_file(patterns_pool_path, db)
    logging.info("patterns pool inizializated")
    helper.get_ontology_from_file(ontology_path, db)
    logging.info("ontology inizializated")


def preprocess_files():
    files = [f for f in os.listdir(texts_path) if os.path.isfile(os.path.join(texts_path, f))]
    print('\ntry to find unprocessed text')
    for file in tqdm(files):
        if db['processed_files'].find({'name': file}).count() != 0:
            logging.info('File [%s] is already in database, skipping' % file)
            continue
        file_path = texts_path + '/' + file
        helper.process_sentences_from_file(file_path, db)
        db['processed_files'].insert({'name': file})
        logging.info('File [%s] was sucessfully added to database' % file)


def main():
    connect_to_database()
    inizialize()

    preprocess_files()

    helper.build_category_index(db)

    text_preprocesser.process_text_for_instances()

    text_preprocesser.process_text_for_patterns(3)

    treshold = 50
    for iteration in range(1, 11):
        print('Iteration [%s] begins' % str(iteration))
        logging.info('=============ITERATION [%s] BEGINS=============' % str(iteration))
        helper.extract_instances(db, iteration)
        helper.evaluate_instances(db, treshold, iteration)
        helper.extract_patterns(db, iteration)
        helper.evaluate_patterns(db, treshold, iteration)
        helper.zero_coocurence_count(db)


if __name__ == "__main__":
    main()

import pandas as pd
from pymystem3 import Mystem
import nltk
import string
import pymorphy2
import logging
import pymongo
import pickle

mystem = Mystem()
punctuation = string.punctuation

morph = pymorphy2.MorphAnalyzer()
category_pattern_dict = dict()
INF = 100 * 100 * 100

MODE = 1


def get_patterns_from_file(file, db):
    logging.info('Extracting initial patterns from file')
    file = pd.read_excel(file)
    for index, row in file.iterrows():
        if db['patterns'].find({'_id': int(row['id'])}).count() != 0:
            continue
        pattern = dict()
        pattern['_id'] = int(row['id'])
        pattern['string'] = row['pattern']

        arg1, arg2 = dict(), dict()
        arg1['case'] = row['arg1_case'].lower()
        arg1['num'] = row['arg1_num'].lower()
        arg1['pos'] = row['arg1_pos'].lower()
        arg2['case'] = row['arg2_case'].lower()
        arg2['num'] = row['arg2_num'].lower()
        arg2['pos'] = row['arg2_pos'].lower()

        pattern['arg1'] = arg1
        pattern['arg2'] = arg2

        pattern['presicion'] = INF
        pattern['true_detective'] = 0
        pattern['false_detective'] = 0
        # -1 in extracted category id in case when it's our initial pattern
        pattern['extracted_category_id'] = -1
        pattern['used'] = True
        pattern['coocurence_count'] = INF

        # FIXME think about this features more deeply later
        pattern['iteration_added'] = list()
        pattern['iteration_deleted'] = list()

        db['patterns'].insert(pattern)


def get_ontology_from_file(file, db):
    print('Extracting initial ontology from file')
    file = pd.read_excel(file)
    for index, row in file.iterrows():
        ontology_category = dict()
        category_name = mystem.lemmatize(row['categoryName'])[0]

        if db['ontology'].find({'category_name': category_name}).count() != 0:
            continue
        ontology_category['category_name'] = category_name
        ontology_category['_id'] = db['ontology'].find().count() + 1
        if type(row['seedInstances']) is float:
            ontology_category['instances'] = list()
        else:
            ontology_category['instances'] = row['seedInstances'].split('"')[1::2]

        if type(row['seedExtractionPatterns']) is float:
            ontology_category['extraction_patterns'] = list()
        else:
            ontology_category['extraction_patterns'] = [int(s) for s in row['seedExtractionPatterns'].split(' ') if
                                                        s.isdigit()]

        ontology_category['promoted_patterns'] = list()

        for instance in ontology_category['instances']:
            promoted_instance = dict()
            promoted_instance['lexem'] = instance
            promoted_instance['_id'] = db['promoted_instances'].find().count() + 1
            promoted_instance['category_name'] = category_name
            promoted_instance['used'] = True
            promoted_instance['precision'] = 1.0
            promoted_instance['extracted_pattern_id'] = -1
            promoted_instance['iteration_added'] = [0]
            promoted_instance['iteration_deleted'] = list()
            # this instances would have the highest precision because was added by default
            promoted_instance['count_in_text'] = INF
            db['promoted_instances'].insert(promoted_instance)

        db['ontology'].insert(ontology_category)


def process_sentences_from_file(file, db):
    text = open(file, 'r').read()
    sentences = nltk.sent_tokenize(text)
    for s in sentences:
        sentence = dict()
        sentence['_id'] = db['sentences'].find().count() + 1
        sentence['string'] = s
        sentence['words'] = list()

        words = nltk.word_tokenize(s)
        for word in words:
            word_dict = dict()
            if word == '[[' or word == ']]' or word == '[' or word == ']' or word == '==' or word == '|' or word == '=':
                continue
            word_dict['original'] = word
            if word in punctuation:
                word_dict['punctuation'] = True
                word_dict['lexem'] = word
                sentence['words'].append(word_dict)
                continue

            p = morph.parse(word)
            word_dict['pos'] = p[0].tag.POS
            word_dict['case'] = p[0].tag.case
            word_dict['lexem'] = p[0].normal_form
            word_dict['number'] = p[0].tag.number
            word_dict['punctuation'] = False
            sentence['words'].append(word_dict)

        db['sentences'].insert(sentence)
    return


def build_category_index(db):
    # build indexes on existing ontology in existing database sentences and ontology collections
    print('Building indexes')
    categories = db['ontology'].find()

    for category in categories:
        index = dict()
        index['sentences_id'] = list()
        sentences = db['sentences'].find()
        for sentence in sentences:
            for word in sentence['words']:
                if word['lexem'] == category['category_name']:
                    index['sentences_id'].append(sentence['_id'])
        index['sentences_id'] = list(set(index['sentences_id']))
        index['_id'] = db['indexes'].find().count() + 1
        index['category_name'] = category['category_name']
        db['indexes'].insert(index)
        print('Indexes for category %s are builded' % category['category_name'])
    return


def extract_instances(db, iteration):
    # iterate throw sentences, that contains categories
    # try to find patterns in this sentences
    logging.info("Begin instances extracting")
    category_pattern_dict.clear()
    categories = db['indexes'].find(timeout=False)
    for category in categories:
        for sentence_id in category['sentences_id']:
            sentence = db['sentences'].find_one({'_id': sentence_id})
            patterns = db['patterns'].find({'used': True})
            for pattern in patterns:
                if not (pattern['extracted_category_id'] == -1 or pattern['extracted_category_id'] == category['_id']):
                    continue
                pattern_words_list = nltk.word_tokenize(pattern['string'])
                if ')' in pattern_words_list:
                    pattern_words_list.remove(')')
                arg1_pos, arg2_pos = check_if_pattern_exists_in_sentence(sentence, pattern_words_list)
                if arg1_pos is not None and arg2_pos is not None:
                    arg1 = sentence['words'][arg1_pos]
                    arg2 = sentence['words'][arg2_pos]

                    if arg1['lexem'] == category['category_name'] or \
                                    arg2['lexem'] == category['category_name']:
                        if arg2['lexem'] == category['category_name']:
                            (arg1, arg2) = (arg2, arg1)
                    else:
                        continue

                    if check_words_for_pattern(arg1, arg2, pattern):
                        item = db['promoted_instances'].find({'category_name': category['category_name'],
                                                              'lexem': arg2['lexem']})
                        if item.count() > 0:
                            item = db['promoted_instances'].find_one({'category_name': category['category_name'],
                                                                      'lexem': arg2['lexem']})
                            count_in_text = item['count_in_text']
                            if count_in_text == 0 or count_in_text is None:
                                count_in_text = 1
                            else:
                                count_in_text += 1
                            db['promoted_instances'].update({'_id': item['_id']},
                                                            {'$set': {'count_in_text': count_in_text}})
                            logging.info(
                                'Found excisting instance [%s] for category [%s], with pattern [%s] and [%d] coocurences' % \
                                (arg2['lexem'], category['category_name'], pattern['string'], count_in_text))

                        else:
                            promoted_instance = dict()
                            promoted_instance['_id'] = db['promoted_instances'].find().count() + 1
                            promoted_instance['lexem'] = arg2['lexem']
                            promoted_instance['category_name'] = category['category_name']
                            promoted_instance['used'] = False
                            promoted_instance['precision'] = 0
                            promoted_instance['extracted_pattern_id'] = pattern['_id']
                            promoted_instance['count_in_text'] = 1
                            promoted_instance['iteration_added'] = list()
                            promoted_instance['iteration_added'].append(iteration)
                            promoted_instance['iteration_deleted'] = list()

                            db['promoted_instances'].insert(promoted_instance)
                            logging.info("Found new promoted instance [%s] for category [%s], with pattern [%s]" % \
                                         (promoted_instance['lexem'], category['category_name'], pattern['string']))
    return


def evaluate_instances(db, treshold, iteration):
    logging.info('Begin instances evaluating')
    promoted_instances = db['promoted_instances'].find()
    text_ngrams_dictionary = load_dictionary('ngrams_dictionary_for_instances.pkl')

    for instance in promoted_instances:
        if instance['extracted_pattern_id'] != -1:
            if instance['count_in_text'] == 0:
                continue
            try:
                precision = instance['count_in_text'] / text_ngrams_dictionary[instance['lexem'].lower()]
            except:
                logging.error('Cannot find words %s in ngrams dictionary for instances' % instance['lexem'])
                precision = 0
            logging.info("Precision for promoted instance [%s] for category [%s] updated from [%s] to [%s]" % \
                         (instance['lexem'],
                          instance['category_name'],
                          str(instance['precision']),
                          str(precision)))
            db['promoted_instances'].update({'_id': instance['_id']},
                                            {'$set': {'precision': precision}})

    # for each category we want to have n = 0..20 (will select later) numbers of promoted instances
    # at each iteration we calculate first 20 by precision
    # all that will be out of 20 but was at list earlier will be deleted
    categories = db['ontology'].find(timeout=False)
    for category in categories:
        size = treshold
        promoted_instances_for_category = db['promoted_instances'].find({
            'category_name': category['category_name']}).sort('precision', pymongo.DESCENDING)

        new_instances = 0
        deleted_instances = 0
        stayed_instances = 0
        for promoted_instance in promoted_instances_for_category:
            if promoted_instance['extracted_pattern_id'] == -1:
                stayed_instances += 1
                continue
            # first [n] NOT INITIAL instances must be added
            if size > 0:
                if promoted_instance['used']:
                    logging.info("Promoted instance [%s] stayed for category [%s] with precision [%s]" % \
                                 (promoted_instance['lexem'],
                                  promoted_instance['category_name'],
                                  str(promoted_instance['precision'])))
                    stayed_instances += 1
                else:
                    logging.info("Promoted instance [%s] added for category [%s] with precision [%s]" % \
                                 (promoted_instance['lexem'],
                                  promoted_instance['category_name'],
                                  str(promoted_instance['precision'])))
                    new_instances += 1
                    try:
                        iteration_added = promoted_instance['iteration_added']
                    except:
                        iteration_added = list()
                    iteration_added.append(iteration)
                    db['promoted_instances'].update({'_id': promoted_instance['_id']},
                                                    {'$set': {'used': True,
                                                              'iteration_added': iteration_added}})
                size -= 1

            # other instances must be deleted if they are not in first [n]
            else:
                if promoted_instance['used']:
                    logging.info("Promoted instance [%s] deleted for category [%s] with precision [%s]" % \
                                 (promoted_instance['lexem'],
                                  promoted_instance['category_name'],
                                  str(promoted_instance['precision'])))
                    deleted_instances += 1
                    try:
                        iteration_deleted = promoted_instance['iteration_added']
                    except:
                        iteration_deleted = list()
                    iteration_deleted.append(iteration)
                    db['promoted_instances'].update({'_id': promoted_instance['_id']},
                                                    {'$set': {'used': False,
                                                              'iteration_deleted': iteration_deleted}})

        logging.info("Add [%s] new instances, delete [%s], stayed [%d] instances for category [%s]" % \
                     (str(new_instances), str(deleted_instances), stayed_instances, category['category_name']))

    return


def check_if_pattern_exists_in_sentence(sentence, pattern_words_list):
    # check if pattern is in the sentence and return arg1/arg2 positions
    # FIXME now look into only one-word arguments, need to extend
    pattern_words_list.remove('arg1')
    pattern_words_list.remove('arg2')
    arg1_pos, arg2_pos = None, None

    for i in range(0, (len(sentence['words']) - len(pattern_words_list)) + 1):
        flag = True
        for j in range(0, len(pattern_words_list)):
            if sentence['words'][i]['original'] != pattern_words_list[j]:
                flag = False
                break
            i += 1
        if not flag:
            continue

        arg1_pos = i - len(pattern_words_list) - 1
        arg2_pos = arg1_pos + len(pattern_words_list) + 1
        break

    return (arg1_pos, arg2_pos)


def check_words_for_pattern(arg1, arg2, pattern):
    # check if arguments parameters are the same as in pattern
    try:
        if arg1['case'] == pattern['arg1']['case'] and \
                (arg1['number'] == pattern['arg1']['num'] or pattern['arg1']['num'] == 'all') and \
                        arg1['pos'].lower() == pattern['arg1']['pos'].lower() and \
                        arg2['case'] == pattern['arg2']['case'] and \
                (arg2['number'] == pattern['arg2']['num'] or pattern['arg2']['num'] == 'all') and \
                        arg2['pos'].lower() == pattern['arg2']['pos'].lower():
            return True
    except:
        pass
    return False


def extract_patterns(db, iteration=1):
    logging.info('Begin pattern extraction')
    categories = db['indexes'].find(timeout=False)
    for category in categories:
        for sentence_id in category['sentences_id']:
            instances = db['promoted_instances'].find({'category_name': category['category_name'],
                                                       'used': True})
            sentence = db['sentences'].find_one({'_id': sentence_id})
            if (sentence[
                    'string'] == 'Toyota Avalon — полноразмерный автомобиль выпускающийся компанией Toyota с 1995 года.'):
                x = 1001
            for instance in instances:
                if check_word_in_sentence(sentence, instance['lexem']) != -1:
                    arg1_pos = check_word_in_sentence(sentence, category['category_name'])
                    arg2_pos = check_word_in_sentence(sentence, instance['lexem'])

                    if abs(arg1_pos - arg2_pos) >= 5:
                        # choose the patterns not more than 5 words im sum with arg1/arg2
                        continue

                    # just because we have different types of patterns we need to check this conditions
                    # to form the pattern string
                    if arg1_pos < arg2_pos:
                        pattern_string = 'arg1 '
                        for i in range(arg1_pos + 1, arg2_pos):
                            pattern_string += sentence['words'][i]['original']
                            pattern_string += ' '
                        pattern_string += 'arg2'
                    else:
                        pattern_string = 'arg2 '
                        for i in range(arg2_pos + 1, arg1_pos):
                            pattern_string += sentence['words'][i]['original']
                            pattern_string += ' '
                        pattern_string += 'arg1'

                    # FIXME add this case when finding instances
                    if '(' in pattern_string and ')' not in pattern_string:
                        pattern_string += ' )'

                    if pattern_string == 'arg1 arg2' or pattern_string == 'arg2 arg1':
                        continue

                    promoted_pattern = dict()
                    promoted_pattern['arg1'] = dict()
                    promoted_pattern['arg1']['num'] = sentence['words'][arg1_pos]['number']
                    promoted_pattern['arg1']['case'] = sentence['words'][arg1_pos]['case']
                    promoted_pattern['arg1']['pos'] = sentence['words'][arg1_pos]['pos']

                    promoted_pattern['arg2'] = dict()
                    promoted_pattern['arg2']['num'] = sentence['words'][arg2_pos]['number']
                    promoted_pattern['arg2']['case'] = sentence['words'][arg2_pos]['case']
                    promoted_pattern['arg2']['pos'] = sentence['words'][arg2_pos]['pos']

                    # TODO also need to check arg1/arg2 conditions
                    if db['patterns'].find({'string': pattern_string,
                                            'extracted_category_id': category['_id'],
                                            'arg1': promoted_pattern['arg1'],
                                            'arg2': promoted_pattern['arg2']}).count() > 0:

                        found_pattern = db['patterns'].find_one({'string': pattern_string,
                                                                 'extracted_category_id': category['_id'],
                                                                 'arg1': promoted_pattern['arg1'],
                                                                 'arg2': promoted_pattern['arg2']})
                        coocurence_count = found_pattern['coocurence_count']
                        coocurence_count += 1
                        db['patterns'].update({'_id': found_pattern['_id']},
                                              {'$set': {'coocurence_count': coocurence_count}})

                        logging.info(
                            'Updating excisting pattern [%s] for category [%s] found for instance [%s] with [%d] coocurences' % \
                            (found_pattern['string'], category['category_name'], instance['lexem'],
                             found_pattern['coocurence_count']))

                    elif db['patterns'].find({'string': pattern_string,
                                              'extracted_category_id': -1}).count() > 0:
                        logging.info('Found initial pattern [%s], skipping' % pattern_string)
                        continue
                    else:
                        promoted_pattern['_id'] = db['patterns'].find().count() + 1
                        promoted_pattern['iteration_added'] = [iteration]
                        promoted_pattern['iteration_deleted'] = list()
                        promoted_pattern['used'] = False
                        promoted_pattern['extracted_category_id'] = category['_id']
                        promoted_pattern['coocurence_count'] = 1
                        promoted_pattern['string'] = pattern_string
                        promoted_pattern['precision'] = 0

                        # FIXME think about this metrics later
                        promoted_pattern['true_detective'] = 0
                        promoted_pattern['false_detective'] = 0

                        # TODO think about the situation, when the pattern found with different 'num' field in words,
                        # TODO but the same conditions for everything else

                        db['patterns'].insert(promoted_pattern)
                        logging.info('Found new pattern [%s] for category [%s] found for instance [%s]' % \
                                     (promoted_pattern['string'], category['category_name'], instance['lexem']))
                        break
    return


def evaluate_patterns(db, treshold, iteration):
    logging.info('Begin patterns evaluation')
    patterns = db['patterns'].find()
    text_ngrams_dictionary = load_dictionary('ngrams_dictionary_for_patterns.pkl')
    for pattern in patterns:
        if pattern['extracted_category_id'] == -1:
            continue
        pattern_string = pattern['string']
        pattern_tokens = nltk.word_tokenize(pattern_string)
        pattern_tokens.remove('arg1')
        pattern_tokens.remove('arg2')
        if ')' in pattern_tokens:
            pattern_tokens.remove(')')
        pattern_string = ''
        for token in pattern_tokens:
            pattern_string += token.lower()
            pattern_string += ' '
        pattern_string = pattern_string[:-1]
        if pattern['coocurence_count'] == 0:
            continue
        try:
            precision = pattern['coocurence_count'] / text_ngrams_dictionary[pattern_string]
        except:
            logging.error('Cannot find words %s in ngrams_dict' % pattern_string)
            precision = 0

        db['patterns'].update({'_id': pattern['_id']},
                              {'$set': {'precision': precision}})

    categories = db['ontology'].find(timeout = False)
    for category in categories:
        size = treshold
        promoted_patterns_for_category = db['patterns'].find({
            'extracted_category_id': category['_id']}).sort('precision', pymongo.DESCENDING)
        new_patterns, deleted_patterns, stayed_patterns = 0, 0, 0
        for promoted_pattern in promoted_patterns_for_category:
            if promoted_pattern['extracted_category_id'] == -1:
                continue
            if size > 0:
                if promoted_pattern['used']:
                    logging.info("Promoted pattern [%s] stayed for category [%s] with precision [%s]" % \
                                 (promoted_pattern['string'],
                                  category['category_name'],
                                  str(promoted_pattern['precision'])))
                    stayed_patterns += 1
                else:
                    logging.info("Promoted pattern [%s] added for category [%s] with precision [%s]" % \
                                 (promoted_pattern['string'],
                                  category['category_name'],
                                  str(promoted_pattern['precision'])))
                    new_patterns += 1
                    try:
                        iteration_added = promoted_pattern['iteration_added']
                    except:
                        iteration_added = list()
                    iteration_added.append(iteration)
                    db['patterns'].update({'_id': promoted_pattern['_id']},
                                          {'$set': {'used': True,
                                                    'iteration_added': iteration_added}})
                db['iteration_' + str(iteration)].insert({'pattern': promoted_pattern['string'],
                                                              'category_name': category['category_name']})
                size -= 1
            else:
                if promoted_pattern['used']:
                    logging.info("Promoted instance [%s] deleted for category [%s] with precision [%s]" % \
                                 (promoted_pattern['string'],
                                  category['category_name'],
                                  str(promoted_pattern['precision'])))
                    deleted_patterns += 1
                    try:
                        iteration_deleted = promoted_pattern['iteration_deleted']
                    except:
                        iteration_deleted = list()
                    db['patterns'].update({'_id': promoted_pattern['_id']},
                                          {'$set': {'used': False,
                                                    'iteration_deleted': iteration_deleted}})
        logging.info("Add [%d] new patterns, delete [%d], stayed [%d] patterns for category [%s]" % \
                     (new_patterns, deleted_patterns, stayed_patterns, category['category_name']))
    return


def load_dictionary(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def check_word_in_sentence(sentence, lexem):
    # help to find the word lexem in the sentence and return its position if it exists
    pos = 0
    for word in sentence['words']:
        if lexem == word['lexem']:
            return pos
        pos += 1
    return -1


def zero_coocurence_count(db):
    logging.info('Reser coocurence counts for instances/patterns')
    instances = db['promoted_instances'].find()
    patterns = db['patterns'].find()

    for instance in instances:
        db['promoted_instances'].update({'_id': instance['_id']},
                                        {'$set': {'count_in_text': 0}})

    for pattern in patterns:
        db['patterns'].update({'_id': pattern['_id']},
                              {'$set': {'coocurence_count': 0}})

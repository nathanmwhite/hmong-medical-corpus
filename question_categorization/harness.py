from gensim.models import KeyedVectors

import os
import sys

pos_tag_interface_path = os.path.expanduser(os.path.join('~',\
                                                         'python_workspace', 'medical_corpus_scripting',\
                                                         'pos_tagger_interface'))
sys.path.append(pos_tag_interface_path)
from POS_Tagger import HmongPOSTagger

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import re

subword_embeddings = KeyedVectors.load('subword_model.h5', mmap='r')
tag_embeddings = KeyedVectors.load('tag_alone_model.h5', mmap='r')
token_embeddings = KeyedVectors.load('word2vec_Hmong_SCH.model', mmap='r')

def load_question_data(filename):
    f = open(filename, 'r')
    data = [w.strip().split(' | ') for w in f.readlines() if '???' not in w and '<' not in w]
    f.close()
    print(len(data))
    return data

data = load_question_data('question_type_training_set.txt')

questions, labels = zip(*data)

def tag_question_data(questions):
    tagger = HmongPOSTagger()
    tokenized_questions = [re.sub('([\?,;])', ' \g<1>', q).split(' ') for q in questions]
    return tokenized_questions, tagger.tag_words(tokenized_questions)

tokenized_questions, tags = tag_question_data(questions)

def split_subword_pos_tags(tags):
    """This function takes tags of type B-NN (subword-POS) and produces separate lists for subword tags
    and POS tags"""
    subword_tags = []
    pos_tags = []
    for sent in tags:
        subword_sent = []
        pos_sent = []
        for word in sent:
            if word == '-PAD-': # -PAD- is interpreted as an unknown word that interferes with padding
                subword = 'B'
                pos = 'FW'
            else:
                subword, pos = word.split('-')
            subword_sent.append(subword)
            pos_sent.append(pos)
        subword_tags.append(subword_sent)
        pos_tags.append(pos_sent)
    return subword_tags, pos_tags

subword_tags, pos_tags = split_subword_pos_tags(tags)

# notes: keras.preprocessing.text.one_hot, text_to_word_sequence
# special pad values are used because Keras Tokenizer does not permit 0 as a value
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_questions)
# automatically converts words to sequences of numbers
sequences = tokenizer.texts_to_sequences(tokenized_questions)
word_pad_value = max(tokenizer.word_index.values()) + 1
word_out_value = word_pad_value + 1

pos_tag_tokenizer = Tokenizer()
pos_tag_tokenizer.fit_on_texts(pos_tags)
pos_sequences = pos_tag_tokenizer.texts_to_sequences(pos_tags)
pos_pad_value = max(pos_tag_tokenizer.word_index.values()) + 1
pos_out_value = pos_pad_value + 1

subword_tag_tokenizer = Tokenizer()
subword_tag_tokenizer.fit_on_texts(subword_tags)
subword_sequences = subword_tag_tokenizer.texts_to_sequences(subword_tags)
subword_pad_value = max(subword_tag_tokenizer.word_index.values()) + 1
subword_out_value = subword_pad_value + 1

# can use label_tokenizer.sequences_to_texts once done
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_sequences = [l[0] for l in label_tokenizer.texts_to_sequences(labels)]

MAX_LENGTH = max(len(s) for s in sequences)

word_index = tokenizer.word_index

padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', value=word_pad_value)
padded_pos_sequences = pad_sequences(pos_sequences, maxlen=MAX_LENGTH, padding='post', value=pos_pad_value)
padded_subword_sequences = pad_sequences(subword_sequences, maxlen=MAX_LENGTH, padding='post',\
                                         value=subword_pad_value)

from sklearn.feature_extraction.text import CountVectorizer

def join_data(tokenized_questions, subword_tags, pos_tags):
    joined_data = []
    for i, q in enumerate(tokenized_questions):
        joined_sent = []
        for j, word in enumerate(q):
            joined_sent.append(''.join([word, subword_tags[i][j], pos_tags[i][j]]))
        joined_data.append(' '.join(joined_sent))
    
    print(joined_data[:2])
    return joined_data

#CountVectorizer needs sentences made of strings.
joined_data = join_data(tokenized_questions, subword_tags, pos_tags)

vectorizer = CountVectorizer()
vectorizer.fit(joined_data)
vectors = np.array([v.toarray()[0] for v in vectorizer.transform(joined_data)])

pure_vectorizer_bigrams = CountVectorizer(ngram_range=(2,2))
pure_vectorizer_bigrams.fit(joined_data)
pure_bigram_vectors = np.array([v.toarray()[0] for v in pure_vectorizer_bigrams.transform(joined_data)])

labels_matrix = to_categorical(np.asarray(label_sequences))

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

X_train, X_test, X_vector_train, X_vector_test, X_bigram_vector_train, X_bigram_vector_test, \
X_pos_train, X_pos_test, X_subword_train, X_subword_test,\
y_train, y_test = train_test_split(padded_sequences, vectors,\
                                   pure_bigram_vectors, padded_pos_sequences, padded_subword_sequences,\
                                   label_sequences, test_size=TEST_SIZE)

word_set = set(word_value for sent in X_train for word_value in sent)
pos_set = set(pos_value for sent in X_pos_train for pos_value in sent)
subword_set = set(subword_value for sent in X_subword_train for subword_value in sent)

def create_embeddings_matrix(input_set, index_dict, word2vec_source_object, convert_caps=False):
    '''Creates an embedding matrix from preexisting Word2Vec model for use in the Keras Embedding layer
    @param input_set: the set object containing all unique X entries in X_train as numeral values
    @param index_dict: the Tokenizer.word_index dict object containing numerical conversions
    @param word2vec_source_object: the KeyedVectors object containing the vector values for embedding'''
    # $PAD and $OUT remain zeros; they are max(index_dict.values()) + 1 and + 2, respectively
    pad_out_tags_length = 2
    embedding_matrix = np.zeros((max(index_dict.values()) + pad_out_tags_length + 1,\
                                 word2vec_source_object.vector_size))
    for token, numeral in index_dict.items():
        if numeral in input_set:
            try:
                if convert_caps == True:
                    word2vec_token_value = token.upper()
                else:
                    word2vec_token_value = token
                embedding_vector = word2vec_source_object.wv[word2vec_token_value]
            except KeyError:
                embedding_vector = None
            if embedding_vector is not None:
                embedding_matrix[numeral] = embedding_vector
    return embedding_matrix

words_embedding_matrix = create_embeddings_matrix(word_set, tokenizer.word_index, token_embeddings)
pos_embedding_matrix = create_embeddings_matrix(pos_set, pos_tag_tokenizer.word_index, tag_embeddings, True)
subword_embedding_matrix = create_embeddings_matrix(subword_set, subword_tag_tokenizer.word_index,\
                                                   subword_embeddings, True)

def produce_input_matrix(sequences, embedding_matrix):
    output_sequences = []
    for sent in sequences:
        output_sent = []
        for word in sent:
            output_sent.append(embedding_matrix[word])
        output_sequences.append(output_sent)
    return output_sequences

X_train_sequence_matrix = produce_input_matrix(X_train, words_embedding_matrix)
X_test_sequence_matrix = produce_input_matrix(X_test, words_embedding_matrix)
X_pos_train_sequence_matrix = produce_input_matrix(X_pos_train, pos_embedding_matrix)
X_pos_test_sequence_matrix = produce_input_matrix(X_pos_test, pos_embedding_matrix)
X_subword_train_sequence_matrix = produce_input_matrix(X_subword_train, subword_embedding_matrix)
X_subword_test_sequence_matrix = produce_input_matrix(X_subword_test, subword_embedding_matrix)

y_classes = max(label_tokenizer.word_index.values()) + 1
y_train = to_categorical(y_train, num_classes=y_classes)
y_test = to_categorical(y_test, num_classes=y_classes)

from keras.callbacks import EarlyStopping

def run_model(model, countvectorizer=False, bigrams=False, validation=VALIDATION_SIZE):
    es = EarlyStopping(monitor='accuracy', verbose=1, min_delta=0.01, patience=0)
    if countvectorizer:
        if bigrams:
            model_history = model.fit(np.array(X_bigram_vector_train), y_train,\
         batch_size=4, epochs=50, validation_split=validation, callbacks=[es])
            scores = model.evaluate(np.array(X_bigram_vector_test), y_test)
        else:
            model_history = model.fit(np.array(X_vector_train), y_train,\
         batch_size=4, epochs=50, validation_split=validation, callbacks=[es])
            scores = model.evaluate(np.array(X_vector_test), y_test)
    else:
        model_history = model.fit([X_train_sequence_matrix, X_pos_train_sequence_matrix,\
                                   X_subword_train_sequence_matrix], y_train,\
                                   batch_size=4, epochs=50, validation_split=validation, callbacks=[es])
        scores = model.evaluate([X_test_sequence_matrix, X_pos_test_sequence_matrix,\
                                 X_subword_test_sequence_matrix], y_test)
    print("Training set accuracy: {result:.2f} percent".format(result= \
                                                               model_history.history['accuracy'][-1]*100))
    if validation > 0.0:
        print("Validation set accuracy: {result:.2f} percent".format(result= \
                                                                 model_history.history['val_accuracy'][-1]*100))
    print("Accuracy: {result:.2f} percent".format(result=(scores[1]*100)))

priming_data = load_question_data('priming_set.txt')
priming_questions, priming_labels = zip(*priming_data)
priming_tokenized_questions, priming_tags = tag_question_data(priming_questions)
priming_subword_tags, priming_pos_tags = split_subword_pos_tags(priming_tags)

priming_token_sequences = tokenizer.texts_to_sequences(priming_tokenized_questions)
priming_pos_sequences = pos_tag_tokenizer.texts_to_sequences(priming_pos_tags)
priming_subword_sequences = subword_tag_tokenizer.texts_to_sequences(priming_subword_tags)
priming_label_sequences = [l[0] for l in label_tokenizer.texts_to_sequences(priming_labels)]

priming_y = to_categorical(priming_label_sequences, num_classes=y_classes)

# need to pad these
padded_priming_sequences = pad_sequences(priming_token_sequences,
                                         maxlen=MAX_LENGTH,
                                         padding='post',
                                         value=word_pad_value)
padded_priming_pos_sequences = pad_sequences(priming_pos_sequences,
                                             maxlen=MAX_LENGTH,
                                             padding='post',
                                             value=pos_pad_value)
padded_priming_subword_sequences = pad_sequences(priming_subword_sequences,
                                                 maxlen=MAX_LENGTH,
                                                 padding='post',
                                                 value=subword_pad_value)

# produce_input_matrix
priming_sequence_matrix = produce_input_matrix(padded_priming_sequences,
                                               words_embedding_matrix)
priming_pos_sequence_matrix = produce_input_matrix(padded_priming_pos_sequences,
                                                   pos_embedding_matrix)
priming_subword_sequence_matrix = produce_input_matrix(padded_priming_subword_sequences,
                                                       subword_embedding_matrix)

# need to create countvectorizer versions
priming_joined_data = join_data(priming_tokenized_questions, priming_subword_tags, priming_pos_tags)
priming_vectors = np.array([v.toarray()[0] for v in vectorizer.transform(priming_joined_data)])
priming_bigram_vectors = np.array([v.toarray()[0]\
                                   for v in pure_vectorizer_bigrams.transform(priming_joined_data)])

def prime_model(model, countvectorizer=False, bigrams=False):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    num_epochs = 20
    validation_size = 0.0
    if countvectorizer:
        if bigrams:
            model.fit(np.array(priming_bigram_vectors),
                      priming_y,
                      batch_size=1,
                      epochs=num_epochs)
        else:
            model.fit(np.array(priming_vectors),
                      priming_y,
                      batch_size=1,
                      epochs=num_epochs)
    else:
        model.fit([priming_sequence_matrix, priming_pos_sequence_matrix, priming_subword_sequence_matrix],
                 priming_y, batch_size=1, epochs=num_epochs)
    run_model(model, countvectorizer, bigrams, validation_size)

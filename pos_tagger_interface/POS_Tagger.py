# Hmong POS Tagger
# Copyright (c) 2019 Hmong Medical Corpus Project
# Author: Nathan M. White <nathan.white1@my.jcu.edu.au>
"""
This file contains a POS Tagger based on the Keras POS Tagger model 
associated with the Hmong Medical Corpus Project.
"""

import sys, os

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# Copyright notice
__copyright__ = "Copyright (c) 2019 Hmong Medical Corpus Project"

# project_url
__url__ = "https://corpus.ap-southeast-2.elasticbeanstalk.com/hminterface/"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@my.jcu.edu.au"

# main tagger class
class HmongPOSTagger:
    def __init__(self):
        self.__loc__ = os.path.dirname(__file__)
        self._load_config()
        try:
            self._model = load_model(os.path.join(self.__loc__, 'pos_tagging_model_expanded.h5'))
        except IOError:
            print('IOError: The Keras model file is missing from the current directory!')
        self._populate_tag_indices()
        self._populate_word_indices()
        
    def _load_config(self):
        try:
            with open(os.path.join(self.__loc__, 'pos_tagger.config'), 'r') as f:
                self._config = {a: int(b) for a, b in (w.strip().split('=') for w in f.readlines())}
        except:
            raise
        
    def _retrieve_tabular_data(self, filename):
        '''Retrieves generic tabular data from a text document and returns as a dictionary'''
        try:
            with open(os.path.join(self.__loc__, filename), 'r') as f:
                data = [tuple(w.strip().split('\t')) for w in f.readlines()]
            return {t: int(i) for t, i in data}
        except IOError:
             print('IOError: The required file is missing from the current directory.')
        
    def _populate_tag_indices(self):
        '''Retrieves tags and tag indices from an external text file'''
        self._tag_indices = self._retrieve_tabular_data('bio-pos.txt')
           
    def _populate_word_indices(self):
        '''Retrieves words in vocabulary and word indices from external text file'''
        self._word_indices = self._retrieve_tabular_data('word_indices.txt')
            
    def _convert_words_to_numbers(self, words_list):
        '''Converts words_list strings to numbers based on self._word_indices'''
        numbers_list = []
        for sent in words_list:
            sent_numbers_list = []
            for word in sent:
                try:
                    current_value = self._word_indices[word.lower()]
                except KeyError:
                    current_value = self._word_indices['-OOV-']
                sent_numbers_list.append(current_value)
            numbers_list.append(sent_numbers_list)
        return numbers_list
    
    def _convert_numbers_to_tags(self, tag_numbers_list):
        '''Converts tag indices to tag strings based on self._tag_indices'''
        if not hasattr(self, '_numbers_to_tags'):
            self._numbers_to_tags = {i: t for t, i in self._tag_indices.items()}
        return [[self._numbers_to_tags[np.argmax(i)] for i in sent] for sent in tag_numbers_list]
    
    def _pad_sents(self, sents_list):
        if 'maxlen' not in self._config.keys():
            self._config['maxlen'] = 100
        return pad_sequences(sents_list, maxlen=self._config['maxlen'], padding='post')
    
    def tag_words(self, words_list):
        '''Returns tagged words as a list of tuples of word and tag pairs,
        where @param words_list is a list of strings'''
        try:
            sents_as_nums = self._convert_words_to_numbers(words_list)
        except TypeError:
            raise
            
        #use keras model to tag words using _convert_words_to_numbers, then run model, 
        #    then _convert_numbers_to_tags and return results
        taggable_sents = self._pad_sents(sents_as_nums)
        predicted_as_nums = self._model.predict(taggable_sents)
        results = [[tag for tag in sent[:len(sents_as_nums[i])]] for i, sent \
                   in enumerate(self._convert_numbers_to_tags(predicted_as_nums))]
        
        return results

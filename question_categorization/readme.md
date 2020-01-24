hmong-medical-corpus/question_categorization/

This folder contains code and ancillary materials related to question categorization in Hmong.

__Code files__

The folder contains one code file so far, harness.py. This code file provides the training harness for training a question classification model; it is currently in raw script form, to be updated to a class-based production-grade module (in progress).

__Ancillary files__

1. priming_set.txt : Labeled examples of Hmong question word sequences to assist limited-data training process.
2. question_type_training_set.txt : Main dataset containing labeled questions in Hmong.
3. subword_model.h5 : Word2Vec model for subword positions using BIO notation.
4. tag_alone_model.h5 : Word2Vec model for POS tags.

hmong-medical-corpus/question_categorization/

This folder contains code and ancillary materials related to question categorization in Hmong.

Code files
The folder contains one code file so far, harness.py. This code file provides the training harness for training a question classification model; it is currently in raw script form, to be updated to a class-based production-grade module (in progress).

Ancillary files
priming_set.txt : Labeled examples of Hmong question word sequences to assist limited-data training process.
question_type_training_set.txt : Main dataset containing labeled questions in Hmong.
subword_model.h5 : Word2Vec model for subword positions using BIO notation.
tag_alone_model.h5 : Word2Vec model for POS tags.

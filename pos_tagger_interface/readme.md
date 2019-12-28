/hmong-medical-corpus/pos_tagger_interface/

This folder contains the Python file POS_Tagger.py representing a module containing the Hmong_POS_Tagger class for POS Tagging of Hmong text. In addition to the main file, there are four supporting files:
* pos_tagging_model_expanded.h5 -- the Keras model for the tagging
* bio-pos.txt -- a text file containing the word position/POS labels and their corresponding numerical positions
    in the Keras model's output
* word_indices.txt -- a text file containing the unique word/syllable elements used in the training of the Keras model
    with their corresponding numerical values in the Keras model
* pos_tagger.config -- a configuration file containing additional parameters necessary for the Keras model to work, such as
    the maximum length of sentences for purposes of padding

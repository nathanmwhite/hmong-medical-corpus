#### hmong-medical-corpus/medical_ner_finalized

This folder contains three folders, each containing text files containing medical information in Hmong with one type of medical named entity tag based on UMLS.
Every text file that appears here has been machine annotated and checked by a human expert.

anatomy_tagged : Documents with anatomy-related (BPOC) tags.
disease_tagged : Documents with disease-related (DSYN) tags.
symptom_tagged : Documents with symptom-related (SOSY) tags.

Each text file contains tags according to the following scheme:
1) An initial letter from {B, I, E, O}:
B : beginning of the named entity
I : internal position within the named entity
E : end of the named entity
O : other content
2) A medical named entity tag, based on UMLS tags:
BPOC : anatomy-related (body part or organ, etc.)
DSYN : disease or syndrome
SOSY : sign or symptom

Using Non-speaker Annotations for Low-Resource NER
=====================
Code for paper "[Building Low-Resource NER Models Using Non-Speaker Annotations](https://arxiv.org/abs/2006.09627)", publication TBD

## Abstract
In low-resource natural language processing (NLP), the key problem is a lack of training data in the target language. Cross-lingual methods have had notable success in addressing this concern, but in certain common circumstances, such as insufficient pre-training corpora or languages far from the source language, their performance suffers. In this work we propose an alternative approach to building low-resource Named Entity Recognition (NER) models using "non-speaker" (NS) annotations, provided by annotators with no prior experience in the target language. We recruit 30 participants to annotate unfamiliar languages in a carefully controlled annotation experiment, using Indonesian, Russian, and Hindi as target languages. Our results show that use of non-speaker annotators produces results that approach or match performance of fluent speakers. NS results are also consistently on par or better than cross-lingual methods built on modern contextual representations, and have the potential to further outperform with additional effort. We conclude with observations of common annotation practices and recommendations for maximizing non-speaker annotator performance.

## Reproducing paper results

### Requirements
* Python 3.7
* numpy
* [CogComp-NLPy](https://github.com/CogComp/cogcomp-nlpy)
* [AllenNLP](https://github.com/allenai/allennlp) 0.8.4
* [scikit-learn](https://scikit-learn.org/stable/) 0.22.2 

### Data
Non-speaker (NS) and fluent speaker (FS) manual annotations are provided in the `data/` folder, for each of the three languages used (Indonesian, Russian, Hindi). The `NS-not-empty` directory only contains pre-processed documents that have annotations in them, while the `NS` and `FS` directories may contain documents that annotators were unable to find any annotations in, and hence, are more representitive of the experimental results. 

All annotated documents are stored in JSON format, using the Text Annotation class of [CogComp-NLPy](https://github.com/CogComp/cogcomp-nlpy). To install:
```
pip install cython
pip install ccg_nlpy
```

The `dev` and `test` sets for each use annotated data from [LORELEI Language Packs](https://www.aclweb.org/anthology/L16-1521/), based on a manual split by the authors. 

### Training NER models using [AllenNLP](https://github.com/allenai/allennlp)
We train NER models using the AllenNLP library, and datareaders from the CogComp `ccg` package to be compatible with the Text Annotation data format. Installation instructions for the allennlp virtual environment, library and dependencies are provided in their github repository above. Note that this code has been tested using allennlp version 0.8.4, so you can install this version directly should any errors with the `ccg` package occur.

In order to run the sample script for model training, navigate to the `allennlp` folder and run:
```
./run_sample.sh
```
Update the `run_sample.sh` script directly to: 
1. Change train, dev, test paths
2. Change the random seed used in training

This code assumes you will be training with a GPU - to train without one, edit `configs/ner-mbert.jsonnet` to change the value of "cuda_device" from 0 to -1.

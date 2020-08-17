export RANDOM_SEED=123

CONFIG_FILE=configs/ner-mbert.jsonnet
MODEL_DIR=models/seed$RANDOM_SEED

TRAINING_DIR=../data/ind/train

#TEST_A paths refer to the development directories
IND_TEST_A=../data/ind/dev
RUS_TEST_A=../data/rus/dev
HIN_TEST_A=../data/hin/dev

#TEST_B paths refer to the testing directories
IND_TEST_B=../data/ind/test
RUS_TEST_B=../data/rus/test
HIN_TEST_B=../data/hin/test

export NER_TEST_A_PATH=$IND_TEST_A
export NER_TEST_B_PATH=$IND_TEST_B

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/FS/session1
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NI_session1 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/FS/session2
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NI_session2 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/FS/session3
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NI_session3 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/FS/session4
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NI_session4 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/FS/session5
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NI_session5 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/NS/session1
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NS_session1 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/NS/session2
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NS_session2 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/NS/session3
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NS_session3 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/NS/session4
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NS_session4 --include-package ccg

export NER_TRAIN_DATA_PATH=$TRAINING_DIR/NS/session5
allennlp train $CONFIG_FILE --serialization-dir $MODEL_DIR/ind_NS_session5 --include-package ccg

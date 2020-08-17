from typing import Dict, List, Sequence, Iterable
import os

from overrides import overrides
import ccg_nlpy as ccg

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False


@DatasetReader.register("textannotation_ner")
class TextAnnotationDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenized file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_tags``, ``chunk_tags``, ``ner_tags``.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL 2003 NER data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    """
    _VALID_LABELS = {'ner', 'pos', 'chunk'}

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 coding_scheme: str = "IOB1",
                 sentence_length_threshold: int = -1,
                 remove_empty_sentences: bool = False,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        # this class reads into this scheme.
        self._original_coding_scheme = "IOB1"
        self.remove_empty_sentences = remove_empty_sentences
        
        self.sentence_length_threshold = sentence_length_threshold

    @overrides
    def _read(self, file_paths: str) -> Iterable[Instance]:

        for file_path in file_paths.split(","):
            fnames = os.listdir(file_path)

            for fname in fnames:
                doc = ccg.load_document_from_json(file_path + "/" + fname)
                label_indices = ["O"] * len(doc.tokens)
                if "NER_CONLL" in doc.view_dictionary:
                    ner = doc.get_ner_conll
                if "NER_ONTONOTES" in doc.view_dictionary:
                    ner = doc.get_ner_ontonotes

                if ner is not None:
                    if ner.cons_list is not None:
                        for cons in ner:
                            tag = cons['label']
                            # constituent range end is one past the token
                            for i in range(cons['start'], cons['end']):
                                pref = "I-"
                                # in IOB1: you can't start a sentence with B
                                print(fname)
                                if i not in doc.sentence_end_position and i == cons["start"] and label_indices[i-1][2:] == tag:
                                    pref = "B-"

                                label_indices[i] = pref + tag

                else:
                    print("doc has no ner: ", fname)

                for start, end in zip([0] + doc.sentence_end_position[:-1], doc.sentence_end_position):
                    sent_toks = doc.tokens[start:end]
                    ner_tags = label_indices[start:end]
                    tokens = [Token(token) for token in sent_toks]

                    if -1 < self.sentence_length_threshold < len(tokens):
                        logger.warning("Discarding sentence with length {}".format(len(tokens)))
                        continue

                    if self.remove_empty_sentences and all([t == "O" for t in ner_tags]):
                        continue
                    
                    yield self.text_to_instance(tokens, ner_tags=ner_tags)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         pos_tags: List[str] = None,
                         chunk_tags: List[str] = None,
                         ner_tags: List[str] = None
                         ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        words = [x.text for x in tokens]
        instance_fields: Dict[str, Field] = {'tokens': sequence,
                                             "metadata": MetadataField(
                                                 {"words": words})}

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            coded_chunks = to_bioul(chunk_tags,
                                    encoding=self._original_coding_scheme) if chunk_tags is not None else None
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            # the default IOB1
            coded_chunks = chunk_tags
            coded_ner = ner_tags

        # Add "feature labels" to instance
        if 'pos' in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError("Dataset reader was specified to use pos_tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['pos_tags'] = SequenceLabelField(pos_tags, sequence, "pos_tags")
        if 'chunk' in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError("Dataset reader was specified to use chunk tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['chunk_tags'] = SequenceLabelField(coded_chunks, sequence, "chunk_tags")
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError("Dataset reader was specified to use NER tags as "
                                         " features. Pass them to text_to_instance.")
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, "ner_tags")

        # Add "tag label" to instance
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'pos' and pos_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pos_tags, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'chunk' and coded_chunks is not None:
            instance_fields['tags'] = SequenceLabelField(coded_chunks, sequence,
                                                         self.label_namespace)

        return Instance(instance_fields)

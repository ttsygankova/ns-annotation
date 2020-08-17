from typing import Dict, List, Tuple, Any

from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer

# Not happy about this, but apparently it needs to be done.
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        'flair': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG'
    }
})


@TokenIndexer.register("flair")
class FlairCharIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of character indices.

    Parameters
    ----------
    pretrained_model: ``str``
        The name of the Flair embeddings that you plan to use ("news-forward", for example). These are only
        used here to load the character mapping dictionary and to get the is_forward_lm variable.
    namespace : ``str``, optional (default=``flair``)
        Since Flair uses it's own dictionaries, the vocabulary is actually not used.
    character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    start_chars : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_chars : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    min_padding_length: ``int``, optional (default=``0``)
        We use this value as the minimum length of padding. Usually used with :class:``CnnEncoder``, its
        value should be set to the maximum value of ``ngram_filter_sizes`` correspondingly.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 pretrained_model: str,
                 namespace: str = 'flair',
                 character_tokenizer: CharacterTokenizer = CharacterTokenizer(),
                 start_chars: List[str] = None,
                 end_chars: List[str] = None,
                 min_padding_length: int = 0) -> None:
        self._min_padding_length = min_padding_length
        self._namespace = namespace
        self._character_tokenizer = character_tokenizer

        # by default, start should be "\n" and end should be " " (according to official flair code)
        self._start_chars = [Token(st) for st in (start_chars or ["\n"])]
        self._end_chars = [Token(et) for et in (end_chars or [" "])]

        # really we are just loading this for the character dictionary and the is_forward_lm variable
        # this should match the model used in the embedder.
        flair_embs = FlairEmbeddings(pretrained_model)
        self.dictionary: Dictionary = flair_embs.lm.dictionary
        self.is_forward_lm = flair_embs.is_forward_lm

        # Fixed for now (to match Flair), but could technically be anything...
        self.separator = " "

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            counter[self._namespace][character.text] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:

        indices: List[int] = []

        # don't want to modify tokens for everybody else
        mytoks = list(tokens)
        
        span_start = 0

        for c in self._start_chars:
            indices.append(self.dictionary.get_idx_for_item(c.text))
            span_start += len(c.text)

        spans: List[Tuple[int, int]] = []

        if not self.is_forward_lm:
            # reverse in place. the characters in the words are not reversed though.
            mytoks = mytoks[::-1]

        for i, token in enumerate(mytoks):

            if token.text is None:
                raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
            chars = self._character_tokenizer.tokenize(token.text)
            if not self.is_forward_lm:
                chars = chars[::-1]
            for character in chars:
                index = self.dictionary.get_idx_for_item(character.text)
                indices.append(index)

            # this prevents there being a separator between the last token and the end characters.
            if i < len(mytoks)-1:
                for c in self.separator:
                    indices.append(self.dictionary.get_idx_for_item(c))

            # NOTICE: flair offsets are weird. The beginning offset uses one character *before* the token
            # and the end offset uses one character *after* the token.
            # In practice though, we never actually use the first element of the span because
            # flair uses two separate unidirectional LMs.
            spans.append((span_start-1, span_start+len(token.text)))
            span_start += len(token.text) + len(self.separator)

        if not self.is_forward_lm:
            # when using spans to select tokens from the indices,
            # reversing the spans means that we will select indices
            # the right way around. O_o
            spans = spans[::-1]

        for c in self._end_chars:
            indices.append(self.dictionary.get_idx_for_item(c.text))

        return {index_name: indices, index_name + "_spans": spans}

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {}

    @overrides
    def get_padding_token(self) -> List[int]:
        # following flair code, we use a space as the default padding token.
        return self.dictionary.get_idx_for_item(" ")

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[Any]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:
        out = {}
        for key, val in tokens.items():
            if "spans" in key:
                # spans are padded with tuples instead of integers.
                out[key] = pad_sequence_to_length(val, desired_num_tokens[key], default_value=lambda: (0, 0))
            else:
                # It is EXTREMELY important that the padding be a space, as this is how the CLM was trained.
                out[key] = pad_sequence_to_length(val, desired_num_tokens[key],
                                                  default_value=lambda: self.dictionary.get_idx_for_item(" "))

        # When creating a mask, Allennlp doesn't know the difference between a tensor of *characters*
        # and a tensor of *tokens*. To get around this, we do the same trick that was used in the
        # Bert Indexer and inject a mask here, since the span field holds the token information.
        tok_key = list(filter(lambda s: "spans" in s, tokens.keys()))[0]
        num_toks = len(tokens[tok_key])
        desired_toks = desired_num_tokens[tok_key]
        mask = [1]*num_toks + [0]*(desired_toks - num_toks)
        out["mask"] = mask

        return out

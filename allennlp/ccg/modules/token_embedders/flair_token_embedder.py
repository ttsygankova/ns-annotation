"""
Flair Embedder.
"""
import logging

import torch
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn.util import get_text_field_mask
from flair.embeddings import FlairEmbeddings
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

logger = logging.getLogger(__name__)


class FlairEmbedder(TokenEmbedder):

    def __init__(self, flair_model: FlairEmbeddings) -> None:
        super().__init__()

        self.flair_model = flair_model
        self.pretrain_name = self.flair_model.name
        self.output_dim = flair_model.lm.hidden_size

        for param in self.flair_model.lm.parameters():
            param.requires_grad = False

        # In Flair, every LM is unidirectional going forwards.
        # We always extract on the right side.
        comb_string = "y"

        self.span_extractor = EndpointSpanExtractor(input_dim=self.flair_model.lm.hidden_size, combination=comb_string)

        # Set model to None so it is reloaded in the forward.
        # Have no idea what is happening, but something is modifying the model
        # somewhere between init and forward. Reloading in forward works.
        # self.flair_model = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, input_ids: torch.LongTensor, spans: torch.LongTensor) -> torch.Tensor:

        # Super hack. Model is changed between init and forward and we don't know how.
        # if self.flair_model is None:
        #     self.flair_model = FlairEmbeddings(self.pretrain_name)
        #     for param in self.flair_model.lm.parameters():
        #         param.requires_grad = False

        with torch.no_grad():
            # Doesn't matter what this key is, just needs to be a dict.
            mask = get_text_field_mask({"chars": input_ids})

            # trick: fake spans have a sum that is 0 or less.
            # Look at FlairCharIndexer.pad_token_sequence() to see what the default tuple value is
            mask_spans = (spans.sum(dim=2) > 0).long()

            # Shape: (max_char_seq_len, batch_size)
            batch_second_input_ids = input_ids.transpose(0, 1)
            max_seq_len, batch_size = batch_second_input_ids.shape

            hidden = self.flair_model.lm.init_hidden(batch_size)

            prediction, batch_second_rnn_output, _ = self.flair_model.lm.forward(batch_second_input_ids, hidden)

            # Shape: (batch_size, max_char_seq_len)
            rnn_output = batch_second_rnn_output.transpose(1, 0)

            word_embeddings = self.span_extractor(rnn_output.contiguous(), spans, mask, mask_spans).contiguous()

        # Vestigial, for debugging. Leave it in there.
        #sent = "".join([self.flair_model.lm.dictionary.get_item_for_index(i) for i in input_ids[0]])

        # if "aniaN efiw stisiv" in sent:
        #     print()
        #     print(input_ids)
        #     print(word_embeddings)
        #     exit()

        return word_embeddings


@TokenEmbedder.register("flair-pretrained")
class PretrainedFlairEmbedder(FlairEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'news-forward'),

        If the name is a key in the list of pretrained models at
        https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L834
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    """
    def __init__(self, pretrained_model: str) -> None:

        print("LOADING A MODEL CALLED:", pretrained_model)
        flair_embs = FlairEmbeddings(pretrained_model)

        super().__init__(flair_model=flair_embs)

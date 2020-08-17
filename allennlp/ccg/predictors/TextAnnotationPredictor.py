from overrides import overrides
from typing import List
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter, SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token


@Predictor.register('ta_predictor')
class TextAnnotationPredictor(Predictor):
    """
    This is basically a copy of the SentenceTagger from allennlp. It is
    modified to dump output in a more sensible manner.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # Note: this word splitter is required so that we get character offsets.
        #self._tokenizer = SpacyWordSplitter()
        self.model = model

    def predict(self, sent):
        js = {"sentence": sent}
        return self.predict_instance(self._json_to_instance(js))

    def update_output(self, output):
        if "chars" in output:
            chars = output["chars"]
            preds = output["case_predictions"]

            outstr = []
            for c,p in zip(chars, preds):
                if c == "@@PADDING@@":
                    break
                if p == 1:
                    c = c.upper()
                outstr.append(c)

            output["truecased"] = "".join(outstr)

            del output["case_predictions"]
            del output["chars"]
            del output["class_probabilities"]

        if "logits" in output:
            del output["logits"]
        if "encoded_chars" in output:
            del output["encoded_chars"]
        return output

    @overrides
    def predict_instance(self, sent: Instance) -> JsonDict:
        output = super().predict_instance(sent)

        return self.update_output(output)

    @overrides
    def predict_batch_instance(self, sents: List[Instance]) -> List[JsonDict]:
        outputs = super().predict_batch_instance(sents)
        outputs = [self.update_output(o) for o in outputs]

        return outputs

    def load_line(self, line: str) -> JsonDict:
        """
        This will usually be overridden with use_dataset_reader = True on the command line.
        :param line:
        :return:
        """
        return {"sentence": line}

    def dump_line(self, outputs: JsonDict):
        if "truecased" in outputs:
            return " ".join(outputs["words"]) + "\n" + outputs["truecased"] + "\n\n"
        else:
            return outputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)

        chars = [Token(c) for c in list(sentence)]
        case_labels = ["U" if c.isupper() else "L" for c in sentence]

        char_offsets = [(tok.idx, tok.idx + len(tok.text)) for tok in tokens]

        return self._dataset_reader.text_to_instance(tokens, chars, case_labels, char_offsets)

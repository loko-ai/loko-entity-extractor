import datetime
import random
import re
import textwrap
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import spacy
from spacy.training import Example
from spacy.util import minibatch

from config.app_config import LANG_CODE_MAP
from dao.fitting_dao import FitRegistry
from model.extractors.base_rules_extractors import TargaExtractor, CFExtractor, DateExtractor, EmailExtractor, \
    CurrencyExtractor, TelephoneNumberExtractor, InternationalBankAccountNumberExtractor, PartitaIVAextractor
from model.named_entity import NamedEntity
from utils.conversion_utils import convert_to_spacy_trainable_format
from utils.hf.callbacks import FittingCallback
from utils.logger_utils import stream_logger
from utils.training_utils import TrainerState, TrainerControl

logger = stream_logger(__name__)

class TextEntityExtractor(ABC):
    """
        An abstract base entity extractor that define the interface to implement
    """

    def __init__(self, identifier: str):
        """ Constructor """
        self.identifier = identifier
        self.date_of_creation = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.is_trainable = False
        self.is_trained = False

    @abstractmethod
    def __call__(self, text: str) -> List[dict]:
        pass

    @abstractmethod
    def train(self, train_data, fittng: FitRegistry, **kwargs):
        pass

    @abstractmethod
    def tags(self):
        pass


class RulesTextExtractor(TextEntityExtractor):
    """
        An extractor based on fixed rules
    """

    def __init__(self, identifier: str, **kwargs):
        """ Constructor """
        super().__init__(identifier=identifier)
        self.list_of_extractors = [
            TargaExtractor(),
            CFExtractor(),
            DateExtractor(),
            EmailExtractor(),
            CurrencyExtractor(),
            TelephoneNumberExtractor(),
            InternationalBankAccountNumberExtractor(),
            # BankIdentifierCodeExtractor(),
            PartitaIVAextractor()
        ]

    def __call__(self, text: str) -> List[NamedEntity]:
        """ Extraction """
        diz = {}
        result = []
        for extractor in self.list_of_extractors:
            d = []
            ex = extractor(text)
            if ex:
                ex = list(set(ex))
                for ent in ex:
                    d.append({ent: {"occurrences": text.count(ent),
                                    "indexes": [m.start() for m in re.finditer(re.escape(ent), text)]}})
                diz.update({extractor.field: d})
        for ent_type, values_list in diz.items():
            for item_diz in values_list:
                for k, v in item_diz.items():
                    for start_index in v['indexes']:
                        extracted_entity = NamedEntity(tag=ent_type,
                                                       start_index=start_index,
                                                       end_index=start_index + len(k))
                        extracted_entity.entity = k
                        result.append(extracted_entity)
        return result

    def train(self, train_data, **kwargs):
        """ Train """
        raise NotImplementedError("The method 'train' is not implemented for this non trainable extractor model!")

    def tags(self) -> List[str]:
        """ Get tags """
        return [ext.field for ext in self.list_of_extractors]


class BaseSpacyTextExtractor(TextEntityExtractor, ABC):

    def __init__(self, identifier: str, code: str = None, lang: str = None):
        """ Constructor """
        super().__init__(identifier=identifier)
        self.code = code or self._get_code(lang=lang)
        self._nlp = None
        self.lang = lang

    def _get_code(self, lang: str) -> str:
        try:
            return LANG_CODE_MAP[lang]
        except KeyError:
            raise ValueError("Invalid input parameter \'lang\'!")

    @abstractmethod
    def nlp(self):
        pass

    def _extract(self, text: str) -> List[NamedEntity]:
        if len(text) > self.nlp.max_length:
            batches = textwrap.wrap(text, self.nlp.max_length)
        else:
            batches = [text]
        result = []
        for batch in batches:
            doc = self.nlp(batch)
            for entity in doc.ents:
                extracted_entity = NamedEntity(tag=entity.label_,
                                               start_index=entity.start_char,
                                               end_index=entity.end_char)
                extracted_entity.entity = entity.text
                result.append(extracted_entity)
        return result


class SpacyTextExtractor(BaseSpacyTextExtractor):
    """
        A text entity extractor based on the pre-trained models of the library Spacy
    """

    def __init__(self, **kwargs):
        """ Constructor """
        super().__init__(**kwargs)
        self.is_trainable = False
        self.is_trained = True

    @property
    def nlp(self):
        if not self._nlp:
            self._nlp = spacy.load(self.code, disable=["parser", "tagger"])
            self._nlp.max_length = 1000000
            self.lang = self._nlp.meta['lang']
        return self._nlp

    def __call__(self, text: str) -> List[NamedEntity]:
        """ Extraction """
        return self._extract(text=text)

    def train(self, train_data, **kwargs):
        """ Train """
        raise NotImplementedError("The method 'train' is not implemented for this non trainable extractor model!")

    def tags(self) -> List[str]:
        return self.nlp.meta['labels']['ner']


class TrainableSpacyTextExtractor(BaseSpacyTextExtractor):
    """ A trainable text extractor based on Spacy """

    def __init__(self,
                 extend_pretrained: bool = True,
                 n_iter: int = 100,
                 minibatch_size: int = 500,
                 dropout_rate: float = 0.3,
                 **kwargs):
        """ Constructor """
        self.extend_pretrained = extend_pretrained
        self.n_iter = n_iter
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)
        self.is_trainable = True
        self.is_trained = False
        self.extra_tags = set()
        self.current_epoch = "Not_Trained"

    @property
    def nlp(self):
        if not self._nlp:
            ### load existing spaCy pre-trained model (incremental training) ###
            if self.extend_pretrained:
                nlp = spacy.load(self.code)
                logger.info("Loaded model: " + str(self.code))
                self._nlp = nlp
            ### create blank Language class (training of new model) ###
            else:
                nlp = spacy.blank(self.lang)
                logger.info("Created blank " + str(self.lang) + " model")
                self._nlp = nlp
            self._nlp.max_length = 1000000
            self.lang = self._nlp.meta['lang']
        return self._nlp

    def __call__(self, text: str) -> List[NamedEntity]:
        """ Extraction on text """
        if not self.is_trained:
            err_msg = "It's not possible to extract entities from text with an untrained extractor! " \
                      "Firstly you must to train a model!"
            raise Exception(err_msg)
        return self._extract(text=text)

    def train(self,
              train_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]],
              fitting: FitRegistry,
              ws_client=None,
              callbacks=None,
              **kwargs):
        """
        Training

            Example:
                train_data = [
                    ("Uber blew through $1 million a week", {'entities': [(0, 4, 'ORG')]}),
                    ("Android Pay expands to Canada", {'entities': [(0, 11, 'PRODUCT'), (23, 30, 'GPE')]}),
                    ("Spotify steps up Asia expansion", {'entities': [(0, 8, "ORG"), (17, 21, "LOC")]})
                ]
        """
        callbacks = callbacks or [FittingCallback(fitting, self.identifier, ws_client)]
        state = TrainerState(num_train_epochs=self.n_iter)
        control = TrainerControl()

        for c in callbacks:
            c.model = self
            c.on_train_begin()

        train_data = convert_to_spacy_trainable_format(train_data)
        # create the built-in pipeline components and add them to the pipeline
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        # otherwise, get it so we can add labels
        else:
            ner = self.nlp.get_pipe("ner")

        # add labels
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])
                self.extra_tags.add(ent[2])  # add new labels to extra_tags

        print("New tags add to the model", self.extra_tags)

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]

        # only train NER (disable all the other)
        with self.nlp.disable_pipes(*other_pipes):

            # reset and initialize the weights randomly – but only if we're training a new model
            if not self.is_trained and not self.extend_pretrained:
                self.nlp.begin_training()

            # for each epoch
            for itn in range(self.n_iter):
                state.epoch = itn
                for c in callbacks:
                    c.on_epoch_begin(state=state)
                logger.info("\t >>> Epoch: " + str(itn))
                self.current_epoch = str(itn) + " / " + str(self.n_iter)
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=self.minibatch_size)  # compounding(4.0, 32.0, 1.001)
                i = 0
                for batch in batches:
                    logger.info("\t\t\t\t - Batch number: " + str(i))
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        self.nlp.update([example], losses=losses, drop=self.dropout_rate)
                        for c in callbacks:
                            control = c.on_substep_end(state=state, control=control) or control
                        if control.should_training_stop:
                            break
                    if control.should_training_stop:
                        break
                    i += 1
                    logger.info("\t\t Losses: " + str(losses))
                    logger.info("")
                if control.should_training_stop:
                    break
                for c in callbacks:
                    control = c.on_epoch_end(state=state, metrics=dict(train_loss=losses['ner']), control=control) or control

        self.current_epoch = "Training_Complete"
        self.is_trained = True
        for c in callbacks:
            c.on_train_end()

    def tags(self) -> List[str]:
        if self._nlp:
            if self.is_trained or self.extend_pretrained:
                return self.nlp.meta['labels']['ner']
        return []


if __name__ == '__main__':
    code = 'it_core_news_lg'
    extractor = TrainableSpacyTextExtractor(identifier='123', code=code)
    print(extractor.__dict__)

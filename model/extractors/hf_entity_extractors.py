from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from transformers import BertTokenizerFast, BertForTokenClassification

from config.app_config import MODELS_DIR
from dao.fitting_dao import FitRegistry
from model.client_request import TextRequest, TrainingRequest
from model.extractors.entity_extractors import TextEntityExtractor
from model.named_entity import NamedEntity
from utils.hf.callbacks import compute_metrics, FittingCallback
from utils.hf.dataset import HFdataset
from utils.hf.trainer import CustomTrainingArguments, CustomTrainer, custom_data_collator
from utils.logger_utils import stream_logger

logger = stream_logger(__name__)


class BaseHFTextExtractor(TextEntityExtractor, ABC):

    def __init__(self, identifier: str, code: str = None, lang: str = None, max_len: int = 128,
                 training_args: dict = None, **kwargs):
        """ Constructor """
        super().__init__(identifier=identifier)
        self.code = code or self._get_code(lang=lang)
        self._nlp = None
        self.lang = lang
        self.max_len = max_len
        self._args = dict(save_strategy='no', no_cuda=True, use_legacy_prediction_loop=True,
                          remove_unused_columns=False)
        if training_args:
            self._args.update(training_args)

    def _get_code(self, lang):
        return 'dbmdz/bert-base-italian-xxl-uncased'

    @abstractmethod
    def nlp(self):
        pass

    def _extract(self, text: str) -> List[NamedEntity]:
        test_data = TextRequest(text=text)

        test_set = HFdataset([test_data],
                             self.nlp['tokenizer'],
                             self.max_len)

        args = CustomTrainingArguments(**self._args)

        trainer = CustomTrainer(data_collator=custom_data_collator,
                                model=self.nlp['model'],
                                args=args,
                                compute_metrics=compute_metrics)

        entities = trainer.predict(test_set).cleaned_predictions
        if entities:
            entities = entities[0]
        res = []
        for entity in entities:
            extracted_entity = NamedEntity(tag=entity['tag'],
                                           start_index=entity['start'],
                                           end_index=entity['end'])
            extracted_entity.entity = entity['text']
            res.append(extracted_entity)

        return res


class HFTextExtractor(BaseHFTextExtractor):
    """
        A text entity extractor based on the pre-trained models of the library HF
    """

    def __init__(self, **kwargs):
        """ Constructor """
        super().__init__(**kwargs)
        self.is_trainable = False
        self.is_trained = True

    def __call__(self, text: str) -> List[NamedEntity]:
        """ Extraction """
        return self._extract(text=text)

    @property
    def nlp(self):
        if not self._nlp:
            tokenizer = BertTokenizerFast.from_pretrained(self.code)
            model = BertForTokenClassification.from_pretrained(self.code)
            self._nlp = dict(model=model, tokenizer=tokenizer)

        return self._nlp

    def tags(self) -> List[str]:
        model = self.nlp['model']
        print(model.config.label2id)
        return sorted(set([l.split('-')[1] if '-' in l else l for l in model.config.label2id if l != 'O']))

    def train(self, data, **kwargs):
        pass


class TrainableHFTextExtractor(BaseHFTextExtractor):
    """ A trainable text extractor based on HF """

    def __init__(self,
                 extend_pretrained: bool = True,
                 n_iter: int = 10,
                 minibatch_size: int = 2,
                 dropout_rate: float = .1,
                 p: float = 1,
                 **kwargs):
        """ Constructor """
        self.extend_pretrained = extend_pretrained
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate
        self.p = p
        self.output_dir = str(Path(MODELS_DIR) / kwargs.get('identifier'))
        training_args = dict(output_dir=self.output_dir, num_train_epochs=n_iter,
                             overwrite_output_dir=True, evaluation_strategy='steps',
                             eval_steps=1000, save_total_limit=1,
                             per_device_train_batch_size=minibatch_size,
                             per_device_eval_batch_size=minibatch_size)
        training_args.update(kwargs.get('training_args', {}))
        kwargs['training_args'] = training_args
        super().__init__(**kwargs)
        self.is_trainable = True
        self.is_trained = False
        self.extra_tags = set()
        self.current_epoch = "Not_Trained"

    @property
    def nlp(self):
        if not self._nlp:
            tokenizer = BertTokenizerFast.from_pretrained(self.code)
            self._nlp = dict(model=None, tokenizer=tokenizer)

        return self._nlp

    def __call__(self, text: str) -> List[NamedEntity]:
        """ Extraction on text """
        if not self.is_trained:
            err_msg = "It's not possible to extract entities from text with an untrained extractor! " \
                      "Firstly you must to train a model!"
            raise Exception(err_msg)
        return self._extract(text=text)

    def train(self,
              train_data: TrainingRequest, fitting: FitRegistry, ws_client=None, **kwargs):
        """
        Training

            Example:
                train_data = [
                    ("Uber blew through $1 million a week", {'entities': [(0, 4, 'ORG')]}),
                    ("Android Pay expands to Canada", {'entities': [(0, 11, 'PRODUCT'), (23, 30, 'GPE')]}),
                    ("Spotify steps up Asia expansion", {'entities': [(0, 8, "ORG"), (17, 21, "LOC")]})
                ]
        """

        # create the built-in pipeline components and add them to the pipeline
        dataset = HFdataset(train_data.training_instances,
                            self.nlp['tokenizer'],
                            self.max_len,
                            p=self.p)

        if not self.nlp['model']:
            self.nlp['model'] = BertForTokenClassification.from_pretrained(self.code,
                                                                           num_labels=len(dataset.labels_to_ids),
                                                                           label2id=dataset.labels_to_ids,
                                                                           id2label=dataset.ids_to_labels,
                                                                           hidden_dropout_prob=self.dropout_rate,
                                                                           attention_probs_dropout_prob=self.dropout_rate)
            if not self.extend_pretrained:
                layers = [layer for part in self.nlp['model'].bert._modules.values() for layer in
                          part._modules.values()]
                for layer in layers:
                    self.nlp['model']._init_weights(layer)

        training_args = CustomTrainingArguments(**self._args)
        trainer = CustomTrainer(data_collator=custom_data_collator,
                                model=self.nlp['model'],
                                args=training_args,
                                train_dataset=dataset,
                                eval_dataset=dataset,
                                compute_metrics=compute_metrics)
        cc = FittingCallback(fitting, self.identifier, ws_client, trainer)
        trainer.add_callback(cc)
        trainer.train()

        self.current_epoch = "Training_Complete"
        self.is_trained = True

    def tags(self) -> List[str]:
        if self._nlp:
            if self.is_trained or self.extend_pretrained:
                return sorted(set([l.split('-')[1] for l in self._nlp['model'].config.label2id if l != 'O']))
        return []


    def __getstate__(self):
        logger.debug("I'm being pickled")
        if self._nlp:
            pass
            # Path(self.output_dir).mkdir(exist_ok=True)
            # tmp = SpooledTemporaryFile()
            # self._nlp['model'].save_pretrained(self.output_dir)
            # model = self._nlp['model'].state_dict()
            # torch.save(model, tmp)
            # NER_DAO._upload(self.identifier+'/'+'pytorch_model.bin', file=tmp)

        res = self.__dict__.copy()
        del res['_nlp']
        logger.debug(res)
        return res

    # def __setstate__(self, d):
    #     root_logger.debug("I'm being unpickled")
    #     output_dir = d['output_dir']
    #     model = None
    #     if d['is_trained']:
    #         model = BertForTokenClassification.from_pretrained(output_dir)
    #     self.__dict__ = d
    #     self._nlp = None
    #     self.nlp['model'] = model
    #     root_logger.debug(self.__dict__)


if __name__ == '__main__':
    extractor = HFTextExtractor(identifier='hello', lang='it')
    s = '''Ordinanza
2) Contributo unificato artt. 16 — 248 Testo Unico

Causa iscritta in data 18/11/2015 tra LUCCHESE UGO/EQUITALIA SUD 1
Omesso Pagamento

L Pagato in modo insufficiente

[C ]Corte di Cassazione
{mporto che deve essere riscosso: euro 43.00 (QUARANTATRE/00)

Domicilio eletto presso l’Avvocato con studio in [LUOGOAVVOCATO] via'''

    print(extractor.tags())

    print([e.__dict__ for e in extractor(s)])

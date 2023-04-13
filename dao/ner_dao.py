import json
import pickle
import shutil
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import List, Dict

from transformers import BertTokenizerFast, BertForTokenClassification

from config.models_config import PRETRAINED_MODELS
from dao.in_memory_dao import InMemoryDAO
from model.extractors.hf_entity_extractors import TrainableHFTextExtractor
from utils.logger_utils import stream_logger

logger = stream_logger(__name__)


class PretrainedNERDao:
    def __init__(self, pretrained_mapping: Dict = PRETRAINED_MODELS):
        self.pretrained_mapping = pretrained_mapping

    def get_all(self) -> List[str]:
        return list(self.pretrained_mapping.keys())

    def load_model(self, identifier: str) -> "TextEntityExtractor":
        if identifier in self.pretrained_mapping:
            logger.info(f'Model {identifier} loaded in memory')
            return self.pretrained_mapping[identifier](identifier=identifier, code=identifier)

    @lru_cache(5)
    def load_blueprint(self, identifier: str) -> Dict:
        ner_model = self.pretrained_mapping[identifier](identifier=identifier, code=identifier)
        bp = dict(type=ner_model.__class__.__name__, tags=ner_model.tags())
        k_to_remove = ["_nlp", "extra_tags", "list_of_extractors"]
        bp.update({k: v for k, v in ner_model.__dict__.items() if k not in k_to_remove})
        return bp


class NERDao(PretrainedNERDao):
    """
        DAO class for save or load a TrainableSpacyTextExtractor object
    """

    def __init__(self, dir_path: str, extension: str = '.pkl', **kwargs):
        self.dir_path = Path(dir_path)
        self.extension = extension
        super().__init__(**kwargs)

    def save(self, identifier: str, ner_model: 'TextEntityExtractor'):
        p = self.dir_path / identifier
        p.mkdir(exist_ok=True)
        try:
            bp = dict(type=ner_model.__class__.__name__, tags=ner_model.tags())
            k_to_remove = ["_nlp", "extra_tags", "list_of_extractors"]
            bp.update({k: v for k, v in ner_model.__dict__.items() if k not in k_to_remove})
            with open(p / 'blueprint.json', 'w') as f:
                json.dump(bp, f, indent=2)

            if isinstance(ner_model, TrainableHFTextExtractor):
                if ner_model._nlp:
                    logger.debug('Save bin')
                    ner_model._nlp['model'].save_pretrained(p)

            fn = p / ('model' + self.extension)
            with open(fn, 'wb') as f:
                pickle.dump(ner_model, f)
            #joblib.dump(value=ner_model, filename=fn)
        except Exception:
            raise Exception("Error in save model!")

    def get_all(self) -> List[str]:
        custom_models = sorted([el.name for el in self.dir_path.glob('*') if el.is_dir()])
        pretrained_models = super().get_all()
        return custom_models + pretrained_models

    def get_all_files(self, identifier: str) -> List[list]:
        if identifier in super().get_all():
            raise Exception("Default models can't be exported")
        files = []
        for el in (self.dir_path/identifier).rglob('*'):
            buffer = BytesIO()
            if not el.is_dir():
                with open(el, 'rb') as f:
                    buffer.write(f.read())
                buffer.seek(0)
                files.append([str(el.relative_to(self.dir_path)), buffer])
        return files

    def save_files(self, files: List[list]):

        for fname, content in files:
            (self.dir_path / fname).parent.mkdir(exist_ok=True, parents=True)
            with open(self.dir_path/fname, 'wb') as f:
                f.write(content.read())

    def load_model(self, identifier: str) -> "TextEntityExtractor":
        if identifier not in self.get_all():
            raise FileNotFoundError("The specified identifier doesn't exist!")
        else:
            try:
                if identifier in super().get_all():
                    return super().load_model(identifier)
                fn = self.dir_path / identifier / ('model' + self.extension)
                with open(fn, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, TrainableHFTextExtractor):
                    # TODO rimetterlo dentro la classe giusta
                    obj._nlp = None
                    if obj.is_trained:
                        model = BertForTokenClassification.from_pretrained(self.dir_path / identifier)
                        obj.nlp['model'] = model
                return obj
                # return joblib.load(filename=fn)
            except Exception:
                raise Exception("Error in load model!")

    def load_model_im(self, identifier: str, im_dao: InMemoryDAO) -> "TextEntityExtractor":
        if identifier in im_dao.objs:
            logger.info(f'Model {identifier} already in memory')
            return im_dao.get_by_id(identifier)
        model = self.load_model(identifier)
        im_dao.load(identifier, model)
        logger.info(f'Model {identifier} loaded in memory')
        return model

    def load_blueprint(self, identifier: str) -> Dict:
        if identifier not in self.get_all():
            raise FileNotFoundError("The specified identifier doesn't exist!")
        else:
            try:
                if identifier in super().get_all():
                    return super().load_blueprint(identifier=identifier)
                fn = self.dir_path / identifier / 'blueprint.json'
                with open(fn, 'r') as f:
                    bp = json.load(f)
                return bp
            except Exception:
                raise Exception("Error in load model!")

    def copy(self, identifier: str, new_name: str):
        if identifier not in self.get_all():
            raise FileNotFoundError("The specified identifier doesn't exist!")
        ner_model = self.load_model(identifier)
        ner_model.identifier = new_name
        self.save(new_name, ner_model)

    def delete(self, identifier: str):
        if identifier in super().get_all():
            raise Exception("Default models can't be deleted")
        if identifier not in self.get_all():
            raise FileNotFoundError("The specified identifier doesn't exist!")
        else:
            try:
                fn = self.dir_path / identifier
                shutil.rmtree(fn)
            except Exception:
                raise OSError("Error in delete model!")

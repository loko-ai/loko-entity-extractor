from typing import List

from config.app_config import IT
from model.named_entity import NamedEntity


# class NERType(Enum):
#     RULES = "rules"
#     SPACY = "spacy"
#     TRAINABLE_SPACY = "trainable_spacy"
#
#
# class NERLang(Enum):
#     IT = "it"            # italian
#     EN = "en"            # english


class TextRequest:
    def __init__(self, text: str):
        self.text = text


class ModelCreationRequest:
    def __init__(self,
                 identifier: str,
                 typology: str,
                 lang: str = IT,
                 extend_pretrained: bool = False,
                 n_iter: int = 10,
                 minibatch_size: int = 2,
                 dropout_rate: float = 0.1,
                 **kwargs):
        self.identifier = identifier
        self.typology = typology
        self.lang = lang
        self.extend_pretrained = extend_pretrained
        self.n_iter = n_iter
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs


class TrainingInstanceRequest:
    def __init__(self,
                 text: str,
                 entities: List[NamedEntity]):
        self.text = text
        self.entities = [NamedEntity(*en) for en in entities]


class TrainingRequest:
    def __init__(self, training_instances: List[TrainingInstanceRequest]):
        self.training_instances = [TrainingInstanceRequest(**inst) for inst in training_instances]


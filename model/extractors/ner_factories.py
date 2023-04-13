from config.app_config import RULES, SPACY, TRAINABLE_SPACY, HF, TRAINABLE_HF
from model.client_request import ModelCreationRequest
from model.extractors.entity_extractors import RulesTextExtractor, SpacyTextExtractor, TrainableSpacyTextExtractor
from model.extractors.hf_entity_extractors import HFTextExtractor, TrainableHFTextExtractor


class NERFactory:
    """
        Factory class for instantiate and return a TextExtractor object
        from a ModelCreationRequest object
    """

    def __call__(self, req: ModelCreationRequest):

        ##### RULES EXTRACTOR #####
        if req.typology == RULES:
            return RulesTextExtractor(identifier=req.identifier)

        ##### SPACY EXTRACTOR #####
        elif req.typology == SPACY:
            return SpacyTextExtractor(identifier=req.identifier,
                                      lang=req.lang)

        ##### TRAINABLE SPACY EXTRACTOR #####
        elif req.typology == TRAINABLE_SPACY:
            return TrainableSpacyTextExtractor(identifier=req.identifier,
                                               lang=req.lang,
                                               extend_pretrained=req.extend_pretrained,
                                               n_iter=req.n_iter,
                                               minibatch_size=req.minibatch_size,
                                               dropout_rate=req.dropout_rate)

        ##### SPACY EXTRACTOR #####
        elif req.typology == HF:
            return HFTextExtractor(identifier=req.identifier,
                                   lang=req.lang)

        ##### TRAINABLE HF EXTRACTOR #####
        elif req.typology == TRAINABLE_HF:
            return TrainableHFTextExtractor(identifier=req.identifier,
                                            lang=req.lang,
                                            extend_pretrained=req.extend_pretrained,
                                            n_iter=req.n_iter,
                                            minibatch_size=req.minibatch_size,
                                            dropout_rate=req.dropout_rate,
                                            **req.kwargs)

        ###### OTHERWISE #####
        else:
            raise Exception("The specified 'typology' in model creation request is not supported!")

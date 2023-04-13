from typing import List, Tuple, Dict

from model.client_request import TrainingRequest


def convert_to_spacy_trainable_format(training_req_obj: TrainingRequest) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
    """
        Function that convert a list of training instance
        to the Spacy format for training NER.

        Example:

            Input = [
                {
                    "text": "Who is Shaka Khan?",
                    "entities": [{"start_index": 7, "end_index": 17, "entity": "Shaka Khan", "tag": "PERSON"}]
                },
                {
                    "text": "I like London and Berlin.",
                    "entities": [{"start_index": 7, "end_index": 13, "entity": "London", "tag": "LOC"},
                                 {"start_index": 18, "end_index": 24, "entity": "Berlin", "tag": "LOC"}]
                }
            ]


            Output =[
                ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
                ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
            ]
    """
    result = []
    for instance in training_req_obj.training_instances:
        c1 = str(instance.text)
        doc_entities = list()
        for ne in instance.entities:
            doc_entities.append(tuple((int(ne.start_index),
                                       int(ne.end_index),
                                       str(ne.tag))))
        c2 = dict()
        c2["entities"] = doc_entities
        result.append(tuple((c1, c2)))
    return result

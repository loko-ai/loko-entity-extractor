from collections import Counter
from datetime import datetime

import spacy

from sklearn.metrics import classification_report, confusion_matrix

from model.named_entity import NamedEntity


def evaluate(ner_model, examples, spacy_model, pretty=True):
    def _add_entities(doc, entities):
        res = doc.copy()
        spans = []
        for ent in entities:
            span = res.char_span(ent['start_index'], ent['end_index'], label=ent['tag'])
            if span:
                spans.append(span)
        res.set_ents(entities=spans)#default="unmodified")
        return res

    nlp = spacy.load(spacy_model)
    y_true = []
    y_pred = []

    for row in examples:
        input_ = row['text']
        annot = [NamedEntity(start,end,tag).__dict__ for start,end,tag in row['entities']]
        pred = [ne.__dict__ for ne in ner_model(text=input_)]
        doc = nlp(input_)
        annotations = _add_entities(doc, annot)
        predictions = _add_entities(doc, pred)
        # print('REFERENCE:')
        # print([(el.text, el.i, el.pos_, el.ent_type_, el.ent_iob_) for el in annotations])
        y_true += [el.ent_type_ or 'O' for el in annotations]
        # print('PREDICTED')
        # print([(el.text, el.i, el.pos_, el.ent_type_, el.ent_iob_) for el in pred])
        y_pred += [el.ent_type_ or 'O' for el in predictions]
    distro = dict(Counter(y_true))
    labels = sorted(set(y_true).union(y_pred))
    cr = classification_report(y_true,y_pred, output_dict=True)
    metrics = dict(accuracy=round(cr['accuracy'],2))
    del cr['accuracy']
    cr = [dict(dict(label=k), **v) for k,v in cr.items()]
    exc = ['label', 'support']
    cr = [dict((k + ' (%)', round(v*100)) if k not in exc else (k, v) for k, v in row.items()) for row in cr]
    cm = dict(labels=labels, values=confusion_matrix(y_true,y_pred).tolist())

    return dict(distro=distro,
                test_report=dict(classification_report=cr, confusion_matrix=cm, metrics=metrics))


if __name__ == '__main__':
    from pprint import pprint

    from dao.ner_dao import NERDao
    from config.app_config import LANG_CODE_MAP

    data = [
              {"text": "Uber blew through $1 million a week", "entities": [[0, 4, "ORG"]]},
              {"text": "Android Pay expands to Canada", "entities": [[0, 11, "PRODUCT"], [23, 29, "GPE"]]},
              {"text": "Spotify steps up Asia expansion", "entities": [[0, 7, "ORG"], [17, 21, "LOC"]]}
        ]

    ner_dao = NERDao(dir_path='../../resources/')
    extractor = ner_dao.load_model('ce')
    spacy_model = LANG_CODE_MAP[extractor.lang]
    # print(data)
    results = evaluate(extractor, data, spacy_model)
    print(results)
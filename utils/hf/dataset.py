import random
import re

import torch
from torch.utils.data import Dataset


class HFdataset(Dataset):
    def __init__(self, data, tokenizer, max_len, p=1):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.p = p
        self.add_entities = False
        self.splitted_data = self._split_data(data)
        if self.add_entities:
            self.labels_to_ids = self._labels2ids(data)
            self.ids_to_labels = {v: k for k, v in self.labels_to_ids.items()}
        self.original_sentences = {i: el.text for i, el in enumerate(data)}

    def _labels2ids(self, data):
        entities = sorted(set([ent.tag for row in data for ent in row.entities]))
        entities = [prefix + ent for ent in entities for prefix in ['B-', 'I-']]
        labels_to_ids = {ent: idx for idx, ent in enumerate(entities + ['O'])}
        return labels_to_ids

    def _tagging2iob(self, encodings, entities):
        labels = ['O'] * len(encodings['offset_mapping'])
        start_m = {t[0]: i for i, t in enumerate(encodings['offset_mapping'][:-1])}
        end_m = {t[1]: i for i, t in enumerate(encodings['offset_mapping'][:-1])}
        for si, ei, tag in entities:
            if si in start_m and ei in end_m:
                for idx in range(start_m[si], (end_m[ei] + 1)):
                    tag_tmp = 'B-' + tag if idx == start_m[si] else 'I-' + tag
                    labels[idx] = tag_tmp
        return labels

    def _add_padding(self, encodings):
        size = len(encodings['input_ids'])

        if size < self.max_len:
            to_add = self.max_len - size
            encodings['input_ids'] += [0] * to_add
            encodings['token_type_ids'] += [0] * to_add
            encodings['attention_mask'] += [0] * to_add
            encodings['offset_mapping'] += [(0, 0)] * to_add
            encodings['labels'] += [-100] * to_add
            encodings['tokens'] += ['[PAD]'] * to_add
        else:
            encodings['input_ids'] = encodings['input_ids'][:self.max_len]
            encodings['token_type_ids'] = encodings['token_type_ids'][:self.max_len]
            encodings['attention_mask'] = encodings['attention_mask'][:self.max_len]
            encodings['offset_mapping'] = encodings['offset_mapping'][:self.max_len]
            encodings['labels'] = encodings['labels'][:self.max_len]
            encodings['tokens'] = encodings['tokens'][:self.max_len]

    def _split_data(self, data):
        splitted_data = []
        self.add_entities = hasattr(data[0], 'entities')
        for i, row in enumerate(data):
            sentence = row.text
            if self.add_entities:
                entities = sorted(row.entities, key=lambda x: x.end_index)
                print('ENTITIES:', entities)
                print([(sentence[e.start_index:e.end_index], e.tag) for e in entities])
            for splitted_sentence, idxs in self._split_sentence(sentence):
                splitted_row = dict(_id=i, text=splitted_sentence, idxs=idxs)
                if self.add_entities:
                    entities, entities_tmp = self._allign_entities(entities, idxs)
                    splitted_row['entities'] = entities_tmp

                    # se non ci sono entita' tengo frase con probabilita' p
                    if entities_tmp or (random.random() <= self.p):
                        splitted_data.append(splitted_row)
                else:
                    splitted_data.append(splitted_row)
        return splitted_data

    def _allign_entities(self, entities, idxs):
        ### allign entities ###
        entities_tmp = []
        for ent in entities.copy():
            if ent.start_index >= idxs[1]:
                continue
            if not (ent.start_index >= idxs[0] or ent.end_index <= idxs[1]):
                break
            ent_tmp = [max(ent.start_index - idxs[0], 0), min(ent.end_index - idxs[0], idxs[1]), ent.tag]
            entities_tmp.append(ent_tmp)
            if ent.end_index <= idxs[1]:
                entities.pop(0)

        return entities, entities_tmp

    def _split_sentence(self, sentence):
        def get_last_token(idx):
            if not encodings['tokens'][idx + 1].startswith('##'):
                return idx
            return get_last_token(idx - 1)

        if not isinstance(sentence, list):
            # ci conserviamo sia la frase che gli indici
            sentence = [sentence, (0, len(sentence))]
        last_s, idxs = sentence
        encodings = self.tokenizer(last_s, return_offsets_mapping=True)
        encodings['tokens'] = [self.tokenizer.decode([_id]) for _id in encodings['input_ids']]
        size = len(encodings['input_ids'])

        if size <= self.max_len:
            yield [last_s, idxs]
        else:
            last_token = get_last_token(self.max_len - 2)
            last_idx = encodings['offset_mapping'][last_token][1]
            yield [last_s[:last_idx], (idxs[0], idxs[0] + last_idx)]
            yield from self._split_sentence([last_s[last_idx:], (idxs[0] + last_idx, last_idx + idxs[1])])

    def __getitem__(self, index):

        sentence = self.splitted_data[index]['text']
        if self.add_entities:
            entities = self.splitted_data[index]['entities']
        _id = self.splitted_data[index]['_id']
        idxs = self.splitted_data[index]['idxs']

        # print(sentence)
        encodings = self.tokenizer(sentence, return_offsets_mapping=True)
        encodings['tokens'] = [self.tokenizer.decode([_id]) for _id in encodings['input_ids']]  ###

        # print('entities:', [(sentence[s:e], t) for s,e,t in entities])
        # print(idxs)
        encodings['labels'] = self._tagging2iob(encodings, entities) if self.add_entities else []
        # print('tokens:', encodings['tokens'])
        # print('entities:', encodings['labels'])
        encodings['labels'] = [self.labels_to_ids[l] for l in encodings['labels']] if self.add_entities else []
        # print('entities:', encodings['labels'])

        self._add_padding(encodings)

        item = dict(input_ids=torch.as_tensor(encodings['input_ids']),
                    attention_mask=torch.as_tensor(encodings['attention_mask']),
                    labels=torch.as_tensor(encodings['labels']) if self.add_entities else None,
                    token_type_ids=torch.as_tensor(encodings['token_type_ids']),
                    tokens=encodings['tokens'],
                    offset_mapping=encodings['offset_mapping'],
                    sentence_id=_id,
                    idxs=idxs
                    )
        return item

    def __len__(self):
        return len(self.splitted_data)

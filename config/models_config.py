import json
from pathlib import Path

import spacy
import transformers

from model.extractors.entity_extractors import SpacyTextExtractor, RulesTextExtractor
from model.extractors.hf_entity_extractors import HFTextExtractor

TRANSFORMERS_CACHE = Path(transformers.utils.default_cache_path)

hf_models = []
# prendiamo modelli indipendendentemente dal task
for p in TRANSFORMERS_CACHE.glob('*.json'):
    with open(p, 'r') as f:
        content = json.load(f)
        name = content['url'].replace('https://huggingface.co/', '').split('/resolve')[0]
        if 'ner' in name.lower():
            hf_models.append(name)
hf_models = sorted(set(hf_models))


PRETRAINED_MODELS = {m: SpacyTextExtractor for m in spacy.info()['pipelines']}
PRETRAINED_MODELS.update({m: HFTextExtractor for m in hf_models})
PRETRAINED_MODELS.update(dict(rules=RulesTextExtractor))

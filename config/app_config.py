import os
from pathlib import Path

MODEL_CACHE_TIMEOUT = int(os.environ.get('MODEL_CACHE_TIMEOUT', 20))

############################## SPACY ##############################
IT = "it"
EN = "en"
EN_ROBERTA = "en_trf"
LANG_CODE_MAP = {
    EN: "en_core_web_sm",
    IT: "it_core_news_lg",
    EN_ROBERTA: "en_core_web_trf"
}
######################################################################

############################## NER TYPOLOGY ##############################
RULES = "rules"
SPACY = "spacy"
TRAINABLE_SPACY = "trainable_spacy"
HF = 'hf'
TRAINABLE_HF = 'trainable_hf'
###########################################################################

MODELS_DIR = "../resources"
Path(MODELS_DIR).mkdir(exist_ok=True)

############################## S3 #########################################

S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')

S3_SECRET_ACCESS_KEY = os.environ.get('S3_SECRET_ACCESS_KEY')

S3_BUCKET = os.environ.get('S3_BUCKET')

AWS_REGION = os.environ.get('AWS_REGION') or 'eu-south-1'
###########################################################################
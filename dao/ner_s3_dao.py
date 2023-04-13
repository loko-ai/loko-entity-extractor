import json
import pickle
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Dict, List

import boto3
import torch
from botocore.exceptions import ClientError
from transformers import BertForTokenClassification, BertConfig

from dao.in_memory_dao import InMemoryDAO
from dao.ner_dao import PretrainedNERDao
from model.extractors.hf_entity_extractors import TrainableHFTextExtractor
from utils.logger_utils import stream_logger

logger = stream_logger(__name__)


def flat(d):
    nd = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                nd[f"{k}_{k2}"] = v2
        else:
            nd[k] = v
    return nd


def f(path, cont):
    l = []
    max_deep = 1 if not path else len(path.split("/")) + 1

    for x in cont:
        x = flat(x)
        if len([y for y in x['Key'].split("/") if y]) == max_deep:
            if path:
                x['Path'] = x['Key']
                x["Key"] = x['Key'].replace(path + "/", "")
            l.append(x)
    return l


class S3NERDao(PretrainedNERDao):
    """
        DAO class for save or load a TrainableSpacyTextExtractor object
    """

    def __init__(self, access_key: str, secret_access_key: str, bucket: str, region_name: str, extension: str = '.pkl',
                 **kwargs):
        self.access_key = access_key
        self.secret_access_key = secret_access_key
        self.bucket = bucket
        self.region_name = region_name
        self.extension = extension
        self.s3_client = None
        self.client_id = None

        super().__init__()

        while not self.s3_client:
            try:
                self.session = boto3.session.Session(aws_access_key_id=self.access_key,
                                                     aws_secret_access_key=self.secret_access_key,
                                                     region_name=self.region_name)
                self.client_id = self.session.client('sts').get_caller_identity().get('Account')
                self.s3_client = self.session.client('s3')
                logger.debug(f"{self.access_key} Connection succeeded")
            except ClientError as e:
                logger.error(e)
                sleep(1)

    def save(self, identifier: str, ner_model: 'TextEntityExtractor'):
        p = Path(identifier)

        try:
            self._upload(str(p))
            bp = dict(type=ner_model.__class__.__name__, tags=ner_model.tags())
            k_to_remove = ["_nlp", "extra_tags", "list_of_extractors"]
            bp.update({k: v for k, v in ner_model.__dict__.items() if k not in k_to_remove})
            tmp = BytesIO()
            tmp.write(bytes(json.dumps(bp, indent=2), 'utf-8'))
            tmp.seek(0)
            self._upload(str(p / 'blueprint.json'), file=tmp)

            logger.debug(f'SAVE NER MODEL {type(ner_model)} {ner_model}')

            if isinstance(ner_model, TrainableHFTextExtractor):
                if ner_model._nlp:
                    logger.debug('Save config')
                    tmp = BytesIO()
                    config = ner_model._nlp['model'].config.to_json_string()
                    tmp.write(bytes(config, 'utf-8'))
                    tmp.seek(0)
                    self._upload(str(p / 'config.json'), file=tmp)

                    logger.debug('Save bin')
                    pytorch_model = ner_model._nlp['model'].state_dict()
                    tmp = BytesIO()
                    torch.save(pytorch_model, tmp)
                    tmp.seek(0)
                    self._upload(str(p / 'pytorch_model.bin'), file=tmp)

            tmp = BytesIO()
            tmp.write(pickle.dumps(ner_model))
            tmp.seek(0)
            self._upload(str(p / ('model' + self.extension)), file=tmp)

        except Exception:
            raise Exception("Error in save model!")

    def get_all(self) -> List[str]:
        custom_models = sorted(self._ls_dirs())
        pretrained_models = super().get_all()
        return custom_models + pretrained_models


    def get_all_files(self, identifier: str) -> List[list]:
        if identifier in super().get_all():
            raise Exception("Default models can't be exported")
        files = []
        for el in self._ls(identifier, recursive=True, only_files=True):
            tmp = BytesIO()
            f = self._download(el)
            tmp.write(f.read())
            tmp.seek(0)
            files.append([el, tmp])
        return files

    def save_files(self, files: List[list]):
        identifier = files[0][0].split('/')[0]
        p = Path(identifier)
        self._upload(str(p))
        for fname, content in files:
            self._upload(fname, content)


    def load_model(self, identifier: str) -> "TextEntityExtractor":
        if identifier not in self.get_all():
            raise FileNotFoundError("The specified identifier doesn't exist!")
        else:
            try:
                if identifier in super().get_all():
                    return super().load_model(identifier)
                p = Path(identifier)
                obj = pickle.load(self._download(str(p / ('model' + self.extension))))

                if isinstance(obj, TrainableHFTextExtractor):
                    # TODO rimetterlo dentro la classe giusta
                    obj._nlp = None
                    if obj.is_trained:
                        config_file = self._download(str(Path(identifier) / 'config.json'))
                        config_dict = json.loads(config_file.read().decode('utf-8'))
                        config = BertConfig(**config_dict)  # .from_json_file(bpath+'config.json')
                        model = BertForTokenClassification.from_pretrained(obj.code, config=config,
                                                                           ignore_mismatched_sizes=True)
                        model.load_state_dict(torch.load(self._download(str(p / 'pytorch_model.bin'))))
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
                fn = Path(identifier) / 'blueprint.json'
                bp = self._download(str(fn))
                bp = json.loads(bp.read().decode('utf-8'))
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
                for el in self._ls(identifier):
                    self._delete_bucket_file(fpath=identifier + '/' + el)
                self._delete_bucket_file(fpath=identifier)
            except Exception:
                raise OSError("Error in delete model!")

    def _ls_dirs(self, path=''):
        try:
            cont = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=path, StartAfter=path, FetchOwner=True,
                                                  Delimiter='/')
            return [prefix['Prefix'][:-1] for prefix in cont['CommonPrefixes']]
        except Exception as inst:
            logger.exception(inst)
            return []

    def _ls(self, path=None, recursive=False, only_files=False):
        path = path or ""
        if self.s3_client:
            try:
                cont = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=path, StartAfter=path, FetchOwner=True)
            except Exception as inst:
                logger.exception(inst)
                return []

            if 'Contents' in cont:
                cont = [{k: (v if k != 'LastModified' else v.strftime("%m/%d/%Y, %H:%M:%S")) for k, v in el.items()}
                        for el in cont['Contents']]
                for el in cont:
                    if el.get('Size') == 0 and el.get('Key').endswith("/"):
                        el['isDir'] = True
                    else:
                        el['isDir'] = False
                if not recursive:
                    cont = f(path, cont)
                if not only_files:
                    return [el['Key'] for el in cont]
                else:
                    return [el['Key'] for el in cont if el['isDir']==False]
            else:
                if not path:
                    return []
                logger.debug(f"bucket : {self.bucket} path: {path}")
                return False
        else:
            raise Exception("Client and/or bucket is not configured")

    def _upload(self, ffname: str, file=None):

        if self.s3_client:
            if file:
                r = self.s3_client.upload_fileobj(file, self.bucket, ffname)
            else:
                r = self.s3_client.put_object(Bucket=self.bucket, Key=(ffname + '/'))
            if not r:
                logger.debug(f"File caricato correttamente nel bucket {self.bucket}")
                return True
            else:
                logger.debug("Failed - Upload Error")
                return False
        else:
            logger.debug("Client and/or bucket is not configured")
            return False

    def _download(self, fname: str):
        f = BytesIO()
        logger.debug(f'DOWNLOAD {self.bucket} {fname}')
        obj = self.session.resource("s3").Bucket(self.bucket).Object(fname)
        obj.download_fileobj(f)
        f.seek(0)
        return f

    def _delete_bucket_file(self, fpath: str):
        try:
            logger.debug(f'DELETE {self.bucket} {fpath}')
            if "." in fpath:
                r = self.s3_client.delete_object(Bucket=self.bucket, Key=fpath)
            else:
                r = self.s3_client.delete_object(Bucket=self.bucket, Key=f"{fpath}/")
            logger.debug(f'RESPONSE: {r}')
            logger.debug("Bucket file was deleted")
        except Exception as inst:
            logger.exception(inst)
            logger.debug("An error occurred deleting the bucket file")
            return False
        return True


if __name__ == '__main__':
    from config.app_config import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_BUCKET, AWS_REGION

    dao = S3NERDao(access_key=S3_ACCESS_KEY, secret_access_key=S3_SECRET_ACCESS_KEY, bucket=S3_BUCKET,
                   region_name=AWS_REGION)
    print(dao._ls('pincopallo_hf', recursive=True, only_files=True))
    # dao.delete_bucket_file(fpath='prova')
    # for el in dao._ls('pincopallo_hf'):
    #     print('QUI:', el)
    #
    # for el in dao._ls_dirs(''):
    #     print('QUI2:', el)

    # content = dao._download('pincopallo_hf/pytorch_model.bin')
    # p = '/home/cecilia/PycharmProjects/ds4biz-entity-extractor/ds4biz_entity_extractor/resources/hf_ner_final/'
    #
    # tmp = BytesIO()
    # with open(p+'pytorch_model.bin', 'rb') as f:
    #     tmp.write(f.read())
    # tmp.seek(0)
    # dao._upload('hf_ner_final/pytorch_model.bin', file=tmp)
    #
    # tmp = BytesIO()
    # with open(p + 'config.json', 'rb') as f:
    #     tmp.write(f.read())
    # tmp.seek(0)
    # dao._upload('hf_ner_final/config.json', file=tmp)

    # p = '/home/cecilia/PycharmProjects/ds4biz-entity-extractor/ds4biz_entity_extractor/resources/prove/'
    #
    # with open(p + 'pytorch_model.bin', 'wb') as f:
    #     f.write(dao._download('hf_ner_final/pytorch_model.bin').read())
    #
    # with open(p + 'config.json', 'wb') as f:
    #     f.write(dao._download('hf_ner_final/config.json').read())

    # p = '/home/cecilia/PycharmProjects/ds4biz-entity-extractor/ds4biz_entity_extractor/resources/hf_ner_final/pytorch_model.bin'
    # with open(p, 'rb') as f:
    # dao._upload('pincopallo_hf/pytorch_model.bin', file=tmp)

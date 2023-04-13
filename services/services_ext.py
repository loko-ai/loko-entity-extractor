import io
import json
import traceback
from urllib.parse import unquote

from loko_client.business.fs_client import FSClient
# from loko_extensions.business.decorators import extract_value_args

import sanic
from sanic import Sanic, Blueprint
from sanic.exceptions import NotFound, SanicException
from sanic_cors import CORS
from sanic_openapi import swagger_blueprint
from sanic_openapi.openapi2 import doc

from utils.loko_extensions.business.decorators import extract_value_args
from business.ws_client import WSClient
from config.app_config import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_BUCKET, AWS_REGION, LANG_CODE_MAP
from dao.fitting_dao import FitRegistry
from dao.in_memory_dao import InMemoryDAO
from dao.ner_dao import NERDao
from dao.ner_s3_dao import S3NERDao
from model.client_request import TextRequest, ModelCreationRequest, TrainingRequest
from model.extractors.ner_factories import NERFactory
from model.extractors.evaluate import evaluate
from utils.logger_utils import stream_logger
from utils.ppom_reader import get_major_minor_version
from utils.sanic_utils import Json
from utils.services_utils import fithelper
from utils.zip_utils import extract_zipfile, make_zipfile

logger = stream_logger(__name__)

SERVICES_PORT = 8080
MODELS_DIR = "../resources"

imdao = InMemoryDAO()
fs_client = FSClient()
ws_client = WSClient(type='ner')

if S3_ACCESS_KEY:
    NER_DAO = S3NERDao(access_key=S3_ACCESS_KEY, secret_access_key=S3_SECRET_ACCESS_KEY, bucket=S3_BUCKET,
                       region_name=AWS_REGION)
else:
    NER_DAO = NERDao(dir_path=MODELS_DIR)

fitting = FitRegistry()


def get_app(name):
    app = Sanic(name)
    swagger_blueprint.url_prefix = "/api"
    app.blueprint(swagger_blueprint)
    return app


name = "entity-extractor"
app = get_app(name)
bp = Blueprint("default", url_prefix="")
app.config.API_DESCRIPTION = "Entity Extractor Swagger"
app.config["API_VERSION"] = get_major_minor_version()
app.config["API_TITLE"] = name
app.config["REQUEST_MAX_SIZE"] = 20000000000
app.config["REQUEST_TIMEOUT"] = 172800

# # gets current directory
# BASE = os.getcwd()
# app.static('/src', BASE + '/src')
# # serve js file for webpack
# app.static('/', BASE)
# app.static('/main.js', './dist/main.js', name='main.js')
CORS(app)
app.static('/web', "/frontend/dist")

@app.route('/web')
async def index(request):
    return await sanic.response.file('/frontend/dist/index.html')

@app.on_request
async def start_service(request):
    """
    Decorator for manage the exceptions and the logs of a service
    """
    dashes = str("-" * 10)
    logger.info(f'{dashes}  Start:  {request.method} {request.path} {dashes}')


@app.on_response
async def end_service(request, response):
    dashes = str("-" * 10)
    if response.status == 200:
        logger.info(f'{dashes}  End (ok):  {request.method} {request.path} {dashes}')
    else:
        logger.info(
            f'{dashes}  End (error): {response.status} - {json.loads(response.body)["error"]} - {request.method} {request.path} {dashes}')


@app.exception(Exception)
async def manage_exception(request, exception):
    if isinstance(exception, SanicException):
        return sanic.json(dict(error=str(exception)), status=exception.status_code)

    e = dict(error=f'{exception.__class__.__name__}: {exception}')
    if isinstance(exception, NotFound):
        return sanic.json(e, status=404)

    logger.error('TracebackERROR: \n' + traceback.format_exc() + '\n\n', exc_info=True)

    if type(exception) == Exception:
        return sanic.json(dict(error=str(exception)), status=500)

    return sanic.json(e, status=500)


@bp.post('/test')
@doc.tag('test')
@doc.summary('TEST')
@extract_value_args()
async def test_service(value, args):
    logger.debug(f'VALUE: {value}')
    logger.debug(f'ARGS: {args}')
    mkv = args.get('multikeyvalue')
    logger.debug(f'{mkv}')
    return sanic.json('test service')

### FRONTEND SERVICES ###

@bp.get('/extractors/')
@doc.tag('frontend')
@doc.summary('It shows all the created NER models (the instances that can be used for training and/or extraction)')
async def all_ners(request):
    return sanic.json(NER_DAO.get_all())

@bp.get('/extractors/<name>')
@doc.tag('frontend')
@doc.summary('It gives information about the specified extractor')
@doc.consumes(doc.String(name="name"), location="path", required=True)
async def info_ner(request, name):
    name = unquote(name)
    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)
    return sanic.json(NER_DAO.load_blueprint(identifier=name))

@bp.post('/extractors/<name>')
@doc.tag('frontend')
@doc.summary('It creates a NER configuration')
@doc.description('''
    Examples
    --------
    body = {"typology": "trainable_spacy",
            "lang": "it",
            "extend_pretrained": true,
            "n_iter": 100,
            "minibatch_size": 500,
            "dropout_rate": 0.2}
''')
@doc.consumes(doc.JsonBody({}), location="body")
@doc.consumes(doc.String(name="name"), location="path", required=True)
async def create_ner(request, name):
    name = unquote(name)
    if name in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" already exists!', status_code=400)
    req = ModelCreationRequest(identifier=name, **request.json)
    logger.info("Create model: " + str(req.__dict__))
    ner_factory = NERFactory()
    extractor = ner_factory(req=req)
    NER_DAO.save(identifier=extractor.identifier, ner_model=extractor)
    logger.info("Save created model: \'" + str(extractor.identifier) + "\'")
    return sanic.json(f'Model "{extractor.identifier}" created!')

@bp.post("/extractors/import")
@doc.tag('frontend')
@doc.summary('Upload an existing extractor')
@doc.consumes(doc.File(name="file"), location="formData", content_type="multipart/form-data", required=True)
async def upload(request):
    file = request.files.get('file')
    print('FILE::', file)

    name = file.name.strip('.zip')

    if name in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" already exist!', status_code=400)

    buffer = io.BytesIO()
    buffer.write(file.body)

    files = extract_zipfile(buffer)
    NER_DAO.save_files(files)

    return sanic.json('Done')

@bp.get("/extractors/<name>/export")
@doc.tag('frontend')
@doc.summary('Download an existing extractor')
@doc.consumes(doc.String(name="name"), location="path", required=True)
async def download(request, name):
    name = unquote(name)

    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)

    file_name = name + '.zip'
    model_files = NER_DAO.get_all_files(name)
    buffer = io.BytesIO()
    make_zipfile(buffer, model_files)
    buffer.seek(0)

    headers = {'Content-Disposition': 'attachment; filename="{}"'.format(file_name)}
    return sanic.response.raw(buffer.getvalue(), headers=headers)

@bp.delete('/extractors/<name>')
@doc.tag('frontend')
@doc.summary('It deletes a NER model given an id')
@doc.consumes(doc.String(name="name"), location="path", required=True)
async def delete_ner(request, name):
    name = unquote(name)
    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)
    logger.info("Removal of model: " + str(name))
    NER_DAO.delete(identifier=name)
    return sanic.json(f'Removal of model "{name}": [OK]')

@bp.get("/extractors/<name>/copy")
@doc.tag('frontend')
@doc.summary('Copy an existing extractor')
@doc.consumes(doc.String(name="new_name"), location="query")
@doc.consumes(doc.String(name="name"), location="path", required=True)
async def copy(request, name):
    name = unquote(name)

    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)

    new_name = request.args.get('new_name', name + '_copy')

    NER_DAO.copy(name, new_name)

    return sanic.json(f'Predictor "{name}" copied in "{new_name}"')


### LOKO SERVICES ###

@bp.post('create')
@doc.tag('CRUD loko services')
@doc.summary('It creates a NER configuration')
@doc.description('''
    Examples
    --------
    body = {
              "value": {
                "typology": "trainable_spacy",
                "lang": "it",
                "extend_pretrained": true,
                "n_iter": 100,
                "minibatch_size": 500,
                "dropout_rate": 0.2
              },
              "args": {
                "new_model_name": "prova"
              }
            }
''')
@doc.consumes(doc.JsonBody({"value": dict, "args": {"new_model_name": str}}), location="body")
@extract_value_args()
async def create_ner(value, args):
    name = args.get('new_model_name')
    if name in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" already exists!', status_code=400)
    req = ModelCreationRequest(identifier=name, **value)
    logger.info("Create model: " + str(req.__dict__))
    ner_factory = NERFactory()
    extractor = ner_factory(req=req)
    NER_DAO.save(identifier=extractor.identifier, ner_model=extractor)
    logger.info("Save created model: \'" + str(extractor.identifier) + "\'")
    return sanic.json(f'Model "{extractor.identifier}" created!')

@bp.post('info')
@doc.tag('CRUD loko services')
@doc.summary('It gives information about the specified extractor')
@doc.description('''
    Examples
    --------
    body = {
              "value": {},
              "args": {
                "model_name": "prova"
              }
            }
''')
@doc.consumes(doc.JsonBody({"value": {}, "args": {"model_name": str}}), location="body")
@extract_value_args()
async def info_ner(value, args):
    name = args.get('model_name') or args.get('new_model_name')
    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)
    return sanic.json(NER_DAO.load_blueprint(identifier=name))

@bp.post('delete')
@doc.tag('CRUD loko services')
@doc.summary('It deletes a NER model given an id')
@doc.description('''
    Examples
    --------
    body = {
              "value": {},
              "args": {
                "model_name": "prova"
              }
            }
''')
@doc.consumes(doc.JsonBody({"value": {}, "args": {"model_name": str}}), location="body")
@extract_value_args()
async def delete_ner(value, args):
    name = args.get('model_name') or args.get('new_model_name')
    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)
    logger.info("Removal of model: " + str(name))
    NER_DAO.delete(identifier=name)
    return sanic.json(f'Removal of model "{name}": [OK]')

@bp.post("import")
@doc.tag('CRUD loko services')
@doc.summary('Upload an existing extractor')
@doc.consumes(Json(name="args"), location="formData", content_type="multipart/form-data")
@doc.consumes(doc.File(name="file"), location="formData", content_type="multipart/form-data")
@extract_value_args(file=True)
async def upload(file, args):
    logger.debug(f'FILE: {file}')
    logger.debug(f'ARGS: {args}')
    file = file[0]
    name = file.name.strip('.zip')

    if name in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" already exist!', status_code=400)

    buffer = io.BytesIO()
    buffer.write(file.body)

    files = extract_zipfile(buffer)
    NER_DAO.save_files(files)

    return sanic.json('Done')

@bp.post("export")
@doc.tag('CRUD loko services')
@doc.summary('Download an existing extractor')
@doc.description('''
    Examples
    --------
    body = {
              "value": {},
              "args": {
                "model_name": "prova"
              }
            }
''')
@doc.consumes(doc.JsonBody({"value": {}, "args": {"model_name": str}}), location="body")
@extract_value_args()
async def download(value, args):
    name = args.get('model_name') or args.get('new_model_name')

    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)

    file_name = name + '.zip'
    model_files = NER_DAO.get_all_files(name)
    buffer = io.BytesIO()
    make_zipfile(buffer, model_files)
    buffer.seek(0)

    return sanic.json(dict(body=buffer.read().decode('cp037'), fname=file_name))


@bp.post('/fit')
@doc.tag('EE loko services')
@doc.summary('It trains a model to perform Named Entity Recognition')
@doc.description('''
    Examples
    --------
    body = {
              "value": [
                    {"text": "Uber blew through $1 million a week", "entities": [[0, 4, "ORG"]]},
                    {"text": "Android Pay expands to Canada", "entities": [[0, 11, "PRODUCT"], [23, 29, "GPE"]]},
                    {"text": "Spotify steps up Asia expansion", "entities": [[0, 7, "ORG"], [17, 21, "LOC"]]}
            ],
              "args": {
                "model_name": "prova"
              }
            }
''')
@doc.consumes(doc.JsonBody({"value": [], "args": {"model_name": str}}), location="body")
@extract_value_args()
async def train_ner(value, args):
    name = args.get('model_name')
    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)
    if name in fitting.all('alive'):
        raise SanicException(f'Extractor "{name}" is already fitting!', status_code=400)

    extractor_model = NER_DAO.load_model_im(identifier=name, im_dao=imdao)
    train_data = TrainingRequest(training_instances=value)
    if not extractor_model.is_trainable:
        raise SanicException("It's not possible to perform a training with the specified untrainable extractor!",
                             status_code=400)
    t = app.loop.create_task(fithelper(name, extractor_model, train_data, fitting, NER_DAO, ws_client))
    fitting.create(name, task=t)
    msg = 'Training start'
    fitting.add(name, msg)
    ws_client.emit(name, msg)
    return sanic.json('Job submitted')


@bp.post('/extract')
@doc.tag('EE loko services')
@doc.summary('It performs Named Entity Recognition with the model associate to the given id')
@doc.description('''
    Examples
    --------
    body = {
              "value": {"text": "Mario Rossi nato il 10 gennaio 1980, codice fiscale RSSMRA80A10H501W, partita iva 86334519757 con veicolo targato BB 000 HH, ha pagato 200.89 euro. Contatti: mariorossi@email.com, tel 3338899047, iban  IT 60 X 0542811101000000123456."},
              "args": {
                "model_name": "prova"
              }
            }
    body = {
          "value": {"text": "Uber blew through $1 million a week"},
          "args": {
            "model_name": "prova"
          }
        }
''')
@doc.consumes(doc.JsonBody({"value": {"text": str}, "args": {"model_name": str}}), location="body")
@extract_value_args()
async def run_ner(value, args):
    name = args.get('model_name')
    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)
    req = TextRequest(**value)
    extractor_model = NER_DAO.load_model_im(identifier=name, im_dao=imdao)
    logger.info("Load model: " + str(name) +
                     " | typology: " + str(extractor_model.__class__.__name__))
    if extractor_model.is_trainable:
        if not extractor_model.is_trained:
            raise SanicException("It's not possible to extract entities from text with an untrained extractor!",
                                 status_code=400)
    named_entites = extractor_model(text=req.text)
    output = [ne.__dict__ for ne in named_entites]
    return sanic.json(dict(entities=output))


@bp.post('/evaluate')
@doc.tag('EE loko services')
@doc.summary('It evaluates a model to perform Named Entity Recognition')
@doc.description('''
    Examples
    --------
    body = {
              "value": [
                    {"text": "Uber blew through $1 million a week", "entities": [[0, 4, "ORG"]]},
                    {"text": "Android Pay expands to Canada", "entities": [[0, 11, "PRODUCT"], [23, 29, "GPE"]]},
                    {"text": "Spotify steps up Asia expansion", "entities": [[0, 7, "ORG"], [17, 21, "LOC"]]}
            ],
              "args": {
                "model_name": "prova"
              }
            }
''')
@doc.consumes(doc.JsonBody({"value": [], "args": {"model_name": str}}), location="body")
@extract_value_args()
async def evaluate_ner(value, args):
    name = args.get('model_name')
    tokenizer = args.get('tokenizer')
    if name not in NER_DAO.get_all():
        raise SanicException(f'Extractor "{name}" does not exist!', status_code=400)
    if name in fitting.all('alive'):
        raise SanicException(f'Extractor "{name}" is already fitting!', status_code=400)

    extractor_model = NER_DAO.load_model_im(identifier=name, im_dao=imdao)

    if extractor_model.is_trainable:
        if not extractor_model.is_trained:
            raise SanicException("It's not possible to extract entities from text with an untrained extractor!",
                                 status_code=400)
    if not tokenizer:
        tokenizer = LANG_CODE_MAP[extractor_model.lang]
    eval = evaluate(extractor_model, value, tokenizer)
    info = NER_DAO.load_blueprint(identifier=name)
    eval['info'] = info
    return sanic.json(eval)


app.blueprint(bp)

app.run("0.0.0.0", port=8080, auto_reload=True)
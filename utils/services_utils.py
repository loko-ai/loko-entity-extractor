import asyncio
import traceback

from dao.fitting_dao import FitRegistry
from dao.ner_dao import NERDao
from model.client_request import TrainingRequest
from model.extractors.entity_extractors import TextEntityExtractor
from utils.logger_utils import stream_logger

logger = stream_logger(__name__)


async def fithelper(name: str, extractor_model: TextEntityExtractor, train_data: TrainingRequest, fitting: FitRegistry,
                    ner_dao: NERDao, ws_client=None):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extractor_model.train, train_data, fitting, ws_client)
        ner_dao.save(identifier=name, ner_model=extractor_model)
        msg = 'Model saved'
        fitting.add(name, msg)
        if ws_client:
            ws_client.emit(name, msg)
        logger.info("Save trained model: " + str(extractor_model))
        fitting.remove(name)
    except Exception as e:
        msg = f'Error: {e}'
        fitting.add(name, msg)
        if ws_client:
            ws_client.emit(name, msg)
        fitting.remove(name)
        logger.error('TracebackERROR: \n' + traceback.format_exc() + '\n\n')

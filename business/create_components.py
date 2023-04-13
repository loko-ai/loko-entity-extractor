from doc.doc import entities_doc
from utils.loko_extensions.model.components import Arg, Component, save_extensions, Input, Output, Dynamic, AsyncSelect, \
    Select

create_input = Input(id='create', service='create', to='create')
create_output = Output(id='create')

info_input = Input(id='info', service='info', to='info')
info_output = Output(id='info')

delete_input = Input(id='delete', service='delete', to='delete')
delete_output = Output(id='delete')

import_input = Input(id='import', service='import', to='import')
import_output = Output(id='import')

export_input = Input(id='export', service='export', to='export')
export_output = Output(id='export')


crud_inputs = [create_input, info_input, delete_input, import_input, export_input]
crud_outputs = [create_output, info_output, delete_output, import_output, export_output]

create_new = Arg(name='create_new', label='Create New', type='boolean', value=False)
new_model_name = Dynamic(name='new_model_name', label='Model Name', dynamicType='text',
                         parent='create_new', condition='{parent}', required=True)
model_name = Dynamic(name='model_name', label='Model Name', dynamicType='asyncSelect',
                     url='http://localhost:9999/routes/ner/extractors',
                     parent='create_new', condition='!{parent}', required=True)

crud_args = [create_new, new_model_name, model_name]
crud = Component(name='NER Management', description='', group='AI',
                 icon='RiToolsFill',
                 args=crud_args,
                 inputs=crud_inputs, outputs=crud_outputs)


model_name = AsyncSelect(name='model_name', label='Model Name',
                         url='http://localhost:9999/routes/ner/extractors',
                         required=True,
                         value='it_core_news_lg')

tokenizer = Select(name='tokenizer', label='Spacy tokenizer',
                   options=['it_core_news_sm', 'en_core_web_sm', 'en_core_web_md', 'it_core_news_lg'],
                   required=False,
                   value='it_core_news_lg',
                   group='Evaluate configuration')

entities_args = [model_name, tokenizer]

fit_input = Input(id='fit', label='fit', service='fit', to='fit')
fit_output = Output(id='fit', label='fit')

extract_input = Input(id='extract', label='extract', service='extract', to='extract')
extract_output = Output(id='extract', label='extract')

evaluate_input = Input(id='evaluate', label='evaluate', service='evaluate', to='evaluate')
evaluate_output = Output(id='evaluate', label='evaluate')

entities = Component(name='NER', description=entities_doc, group='AI', args=entities_args,
                     inputs=[fit_input, extract_input, evaluate_input], outputs=[fit_output, extract_output, evaluate_output],
                     icon='RiFileTextLine', events=dict(type="ner", field="model_name"))





save_extensions([entities, crud])
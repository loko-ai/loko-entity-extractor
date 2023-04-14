<html><p><a href="https://loko-ai.com/" target="_blank" rel="noopener"> <img style="vertical-align: middle;" src="https://user-images.githubusercontent.com/30443495/196493267-c328669c-10af-4670-bbfa-e3029e7fb874.png" width="8%" align="left" /> </a></p>
<h1>Loko - Entity Extractor</h1><br></html>


 **Entity Extractor** is a Loko extension dealing with Named Entity Recognition models. 
 
It provides **Spacy pretrained models** and a <strong>base rule</strong> model which can be used to predict entities without training. Otherwise, in few steps, you can train new models starting from the <a href="https://huggingface.co/tasks/token-classification">**HF**</a> or <a href="https://spacy.io/usage/linguistic-features#named-entities">**Spacy**</a> models in order to predict your custom entities. 

You can `create`, `delete`, `import`, `export` or make a `copy` of you models directly from the **NER GUI**: 

<p align="center"><img src="https://user-images.githubusercontent.com/30443495/232049963-e4bff2e2-df5e-4d97-a512-233f6d475f1e.png" width="80%" /></p>

You can also manage these operations directly in the flow using the **Entities Manager** component:

<p align="center"><img src="https://user-images.githubusercontent.com/30443495/232048555-1d73d637-3d80-47ae-bbea-1c95946e859d.png" width="60%" /></p>

And `fit`, `predict` and `evaluate` models using the **Entities** component:

<p align="center"><img src="https://user-images.githubusercontent.com/30443495/232049279-b56afd50-7fff-47a5-b9d4-d18cfaa427ea.png" width="60%" /></p>

Finally, if you save the evaluate output using the ".eval" extension, you can visualize it into the **NER GUI** clicking on Report:

<p align="center"><img src="https://user-images.githubusercontent.com/30443495/232050350-3aabb499-1d61-4de4-89be-0d12ae8dacb1.png" width="80%" /></p>

## Configuration

In the file *config.json* you can configure environment variables and volumes: 

```
{
  "main": {
    "environment": {
      "SANIC_REQUEST_MAX_SIZE": 20000000000,
      "SANIC_REQUEST_TIMEOUT": 172800,
      "SANIC_RESPONSE_TIMEOUT": 172800
    },
    "volumes": [
      "/var/opt/loko/ner/resources/:/plugin/resources/"
    ],
    "gui": {
      "name": "NER GUI"
    }
  }
}
```
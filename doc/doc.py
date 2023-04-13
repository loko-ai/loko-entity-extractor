entities_doc = '''### Description
The NER component is used to train, predict and evaluate NER models.

It provides Spacy pretrained models and a base rule model which can be used to predict entities without training.

Otherwise, you can train new models starting from the HF or Spacy models in order to predict your custom entities. 

Its configuration and function in the flow are described below to give the user a general understanding of how this can 
be used.

### Configuration

**Model Name** shows a list of available models. 

In order to fit a new model you have to create it first. You can use the **NER GUI** or the **NER Management** 
component.

### Input

**FIT** and **EVALUATE** input require data in the same format. They accept a list of dictionaries representing the 
sentences of your dataset. Each dictionary has two keys: 
*"text"* and *"entities"*. The key *"entities"* contains a list of entities present in the provided sentence. For each
entity they need three elements: *start_index*, *end_index* and *tag*.

Example:  

```json
[
    {"text": "Uber blew through $1 million a week", "entities": [[0, 4, "ORG"]]},
    {"text": "Android Pay expands to Canada", "entities": [[0, 11, "PRODUCT"], [23, 29, "GPE"]]},
    {"text": "Spotify steps up Asia expansion", "entities": [[0, 7, "ORG"], [17, 21, "LOC"]]}
        ]
```

**EXTRACT** input accepts one sentence at a time.

Let's see an example:

```json
{"text": "Spotify steps up Asia expansion"}
```

### Output

**FIT** output will only provide as response: Job submitted. All training checkpoints are shown on the top of you 
component.

**EXTRACT** output returns the list of predicted entities in the provided text.

Example:  

```json
{"entities": [{"tag":"LOC","start_index":17,"end_index":21,"entity":"Asia"}]}
```

**EVALUATE** output returns a classification report of the input data and informations about the model.
You can save the output using the extension ".eval" and visualize you report using the **NER GUI**.
 
'''
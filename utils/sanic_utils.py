from sanic_openapi.openapi2 import doc

class Json(doc.JsonBody):
    def __init__(self, name, fields=None, **kwargs):
        super().__init__()
        self.name = name
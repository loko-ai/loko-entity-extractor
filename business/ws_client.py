from abc import ABC

from loko_client.utils.requests_utils import URLRequest

GATEWAY = 'http://gateway:8080/'

class GatewayClient(ABC):
    """
        An abstract base gateway client
    """

    def __init__(self, gateway=GATEWAY):
        self.u = URLRequest(gateway)

class WSClient(GatewayClient):

    def __init__(self, type: str, gateway=GATEWAY):
        self.type = type
        super().__init__(gateway=gateway)

    def emit(self, name: str, msg: str):
        data = dict(event_name='event_ds4biz',
                    content=dict(msg=msg,
                                 type=self.type,
                                 name=name))
        r = self.u.emit.post(json=data)
        return r.text

if __name__ == '__main__':
    wsclient = WSClient(type='entities')
    print(wsclient.emit('ce', 'hello'))
import datetime
import time
from threading import Timer

from config.app_config import MODEL_CACHE_TIMEOUT


class InMemoryDAO:

    def __init__(self):
        self.objs = {}
        self.cd = {}
        self.timer = {}
        self.expires = {}

    def all(self):
        for k in self.objs:
            yield {'name': k, 'expires': self.expires[k].strftime("%d/%m/%Y, %H:%M:%S")}

    def load(self, name, obj, cd=MODEL_CACHE_TIMEOUT):
        self.objs[name] = obj
        self.cd[name] = cd
        if self.cd[name]:
            self.countdown(name)

    def remove(self, name):
        if name in self.objs:
            del self.objs[name]

    def get_by_id(self, name):
        if self.cd[name]:
            self.countdown(name)
        return self.objs[name]

    def _task(self, name):
        del self.objs[name]

    def countdown(self, name):
        if name in self.timer:
            self.timer[name] = self.timer[name].cancel()
        self.timer[name] = Timer(self.cd[name], self._task, [name])
        self.timer[name].start()
        self.expires[name] = datetime.datetime.now() + datetime.timedelta(seconds=self.cd[name])


if __name__ == '__main__':
    imdao = InMemoryDAO()
    imdao.load('pippo', {'a': 3}, cd=2)
    print(list(imdao.all()))
    time.sleep(1)
    imdao.get_by_id('pippo')
    print(list(imdao.all()))
    time.sleep(1)
    print(imdao.objs)
    time.sleep(1)
    print(imdao.objs)
    print(list(imdao.all()))

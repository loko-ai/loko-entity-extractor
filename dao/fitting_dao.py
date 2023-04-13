from datetime import datetime


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class FitRegistry:
    def __init__(self):
        self.jobs = dict()

    def create(self, _id, task):
        if _id in self.jobs:
            del self.jobs[_id]
        self.jobs[_id] = dict(status='alive', logs=[], task=task, should_training_stop=False)

    def add(self, _id, status):
        self.jobs[_id]['logs'].append({"_id": _id, "date": now(), "status": status})

    def remove(self, _id):
        if _id in self.jobs:
            self.jobs[_id]['status'] = 'not_alive'

    def all(self, status:str=None):
        if not status:
            return list(self.jobs.keys())
        for j in self.jobs:
            if self.jobs[j]['status']==status:
                yield j

    def get_by_id(self, _id):
        return self.jobs.get(_id)
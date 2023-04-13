from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    v = [1 if p == l else 0 for rowp, rowl in zip(predictions, labels) for p, l in zip(rowp, rowl) if l != -100]
    res = sum(v) / len(v)
    return dict(custom_metric=res)


class FittingCallback(TrainerCallback):

    def __init__(self, fitting, name, ws_client=None, trainer=None) -> None:
        super().__init__()
        self._trainer = trainer
        self._fitting = fitting
        self._ws_client = ws_client
        self._name = name

    def on_train_begin(self, args: TrainingArguments = None,
                       state: TrainerState = None,
                       control: TrainerControl = None,
                       **kwargs):
        pass

    def on_substep_end(self, args: TrainingArguments = None,
                       state: TrainerState = None,
                       control: TrainerControl = None,
                       **kwargs):
        if control and self._fitting.jobs[self._name]['should_training_stop']:
            control.should_training_stop = True
        return control

    def on_step_end(self, args: TrainingArguments = None,
                       state: TrainerState = None,
                       control: TrainerControl = None,
                       **kwargs):
        if control and self._fitting.jobs[self._name]['should_training_stop']:
            control.should_training_stop = True
        return control

    def on_epoch_begin(self, args: TrainingArguments = None,
                       state: TrainerState = None,
                       control: TrainerControl = None,
                       **kwargs):
        pass

    def on_epoch_end(self, args: TrainingArguments = None,
                     state: TrainerState = None,
                     control: TrainerControl = None,
                     metrics: dict = None,
                     **kwargs):
        if control.should_training_stop:
            return
        if not metrics:
            metrics = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        msg = f'EPOCH: {int(state.epoch)}/{state.num_train_epochs} - loss: {round(metrics["train_loss"], 3)}'
        self._fitting.add(self._name, msg)
        if self._ws_client:
            self._ws_client.emit(self._name, msg)

    def on_train_end(self, args: TrainingArguments = None,
                       state: TrainerState = None,
                       control: TrainerControl = None,
                       **kwargs):
        pass
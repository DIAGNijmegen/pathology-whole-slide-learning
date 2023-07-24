import torch

from wsilearn.dl.pl.pl_modules import ModuleBase


class SurvModule(ModuleBase):
    def __init__(self, loss=None, epoch_metrics=['cind'], **kwargs):
        """ loss: name of the loss, dictionary(loss=Name,etc) or None """
        if loss is None or len(loss)==0:
            loss = {'name':'coxph'}
        self.loss_name = loss['name']
        # self.cuts = loss.pop('cuts',None).astype(float)

        if self.loss_name not in ['coxph','nll','multisurv']: raise ValueError('unknown loss %s' % self.loss_name)

        super().__init__(loss=loss, epoch_metrics=epoch_metrics, **kwargs)

    def _compute_loss(self, logits, y, phase):
        if not 'train' in phase and len(y)==1 and self.loss_name=='coxph':
            return None #would otherwise always log that loss can not be computed, less important for valid.
        loss = self.criterion(logits, y)
        if loss.view(-1).isnan().any():
            print('nan in loss!')
            return None
        return loss


    def _post_process(self, logits, target=None):
        if logits.dtype==torch.float16:
            logits = logits.to(torch.float32)

        if self.loss_name == 'coxph':
            logits = torch.exp(-logits)
        elif self.loss_name == 'multisurv':
            logits = logits.sigmoid().sum(1)/logits.shape[1] #just add
        else:
            raise ValueError('unknown loss %s' % self.loss_name)
        return logits, target

    def _add_metrics(self, phase):
        metrics = self.metrics
        from wsilearn.dl.models.survival import ConcordanceIndex
        metrics[phase] = {'cind':ConcordanceIndex()}

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        result = self(batch)
        logits = self._get_logits(result)
        return logits
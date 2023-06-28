from typing import List
import numpy as np
from lifelines.utils import concordance_index
from pycox.models.loss import CoxPHLoss
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor
import torch
from torchmetrics import Metric

from wsilearn.dl.torch_utils import to_numpy
from scipy.special import expit


class ConcordanceIndex(Metric):
    is_differentiable = False
    higher_is_better = True
    preds: List[Tensor]
    targets: List[Tensor]

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `ConcordanceIndex` will save all targets and predictions in buffer."
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        if target.shape[1]!=2:
            raise ValueError('expects target with two channels for os and event, not %s' % str(target.shape))
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self, epoch=None, phase=None, out_dir=None):
        preds = torch.concat(self.preds)
        targets = torch.concat(self.targets)
        preds = to_numpy(preds).ravel()
        targets = to_numpy(targets)

        event = targets[:,0]
        os = targets[:,1]
        cind = concordance_index(os, preds, event)
        # if out_dir is not None: #for debugging
        #     out_dir = Path(out_dir)/'cind_debug'
        #     mkdir(out_dir)
        #     out_path = out_dir/f'{phase}_epoch{epoch}.csv'
        #     df = pd.DataFrame(dict(os=os, event=event, preds=preds, cind=[cind]*len(preds)))
        #     df_save(df, out_path)
        return torch.tensor(cind)

class PycoxCoxphLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = CoxPHLoss()
        self.name = 'coxph'

    def forward(self, output, targets):
        event = targets[:,0]
        duration = targets[:,1]
        if len(targets)==1:
            raise ValueError('for coxph batch size has to be > 1')

        if event.max()>1: raise ValueError('expects events max to be 1, not events.max(), probably confused with durations')
        loss =  self.loss(output, duration, event)
        return loss


""" from coxph """
def hazard2surv(hazard: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Transform discrete hazards to discrete survival estimates.
    Ref: LogisticHazard
    """
    return (1 - hazard).add(epsilon).log().cumsum(1).exp()

def outputs_to_surv(outputs:Tensor):
    hazards = outputs.sigmoid()
    surv = hazard2surv(hazards)
    # surv = to_numpy(surv) #[N,bins]
    return surv


def hard2surv_np(hazard, epsilon: float = 1e-7):
    out = (1 - hazard)+epsilon
    out = np.log(out)
    out = np.cumsum(out, 1)
    return np.exp(out)

def outputs_to_surv_np(outputs):

    hazards = expit(outputs)
    surv = hard2surv_np(hazards)
    # surv = to_numpy(surv) #[N,bins]

    return surv

if __name__ == '__main__':
    arr = np.array([[0, 12, 120],[1, 4, 10]])
    out = outputs_to_surv_np(arr)
    print(out)
    out_pt = outputs_to_surv(torch.tensor(arr))
    print(out_pt)
    assert torch.isclose(torch.tensor(out).to(out_pt.dtype), out_pt, atol=1e-7).all()
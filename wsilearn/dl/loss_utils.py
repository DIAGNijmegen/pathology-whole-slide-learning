import torch
import torch.nn

from wsilearn.utils.cool_utils import is_string


def create_loss_fct(name, weight=None, **kwargs):
    ce_names = ['crossentropy', 'cross-entropy', 'cse', 'ce']
    bce_names = ['bcelogits', 'bce_logits']
    if is_string(name):
        if weight is not None:
            print('loss class weighting:', weight)
            if name not in ce_names and name not in bce_names and name!='dice_ce':
                raise ValueError('loss class weighting works only with crossentropy')
            if not torch.is_tensor(weight):
                weight = torch.tensor(weight, dtype=torch.float)
        if name in ce_names:
            criterion = torch.nn.CrossEntropyLoss(weight=weight, **kwargs)
        elif name in bce_names:
            criterion = torch.nn.BCEWithLogitsLoss(weight=weight, **kwargs)
        elif name.lower() in ['mse']:
            criterion = torch.nn.MSELoss(**kwargs)
        elif name.lower() == 'coxph':
            from wsilearn.dl.models.survival import PycoxCoxphLoss
            criterion = PycoxCoxphLoss(**kwargs)
        elif name=='dice':
            from monai.losses import DiceLoss
            criterion = DiceLoss(softmax=True, to_onehot_y=True, **kwargs)
        elif name=='dice_ce':
            from monai.losses import DiceCELoss
            class MyDiceCELoss(DiceCELoss):
                def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                    #call cross entropy directly and dont check input
                    ce_loss = self.cross_entropy(input, target)
                    dice_loss = self.dice(input, target)
                    total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
                    return total_loss

            criterion = MyDiceCELoss(ce_weight=weight, softmax=True, to_onehot_y=True, **kwargs)
        else:
            raise ValueError('unknown loss function name %s' % name)
    else:
        criterion = name
    return criterion


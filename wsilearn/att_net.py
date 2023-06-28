from wsilearn.dl.models.model_blocks import ResBlock
from wsilearn.dl.models.model_utils import *
from wsilearn.dl.pl.pl_modules import ModuleBase
from wsilearn.dl.pool_utils import nms_set
from wsilearn.dl.torch_utils import print_model_summary

class AttGated2d(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, out_dim=1, dropout=0, dropout_type='dropout',
                 main_act='tanh', gate_act='sigm', att_act=None, groups=1, nms=0, nmsinf=False,
                 in_att=None, att_kernel_size=1, sep_conv=False, att_depth=1, **kwargs):
        super().__init__()
        self.short_name = ''
        self.name = 'gated'
        self.out_dim = out_dim
        self.nms = nms
        self.nmsinf = nmsinf
        if nms==1: raise ValueError('nms must not be 1')
        elif nms>1 and nms % 2 ==0: raise ValueError('nms must be odd')

        if len(str(att_kernel_size))==1: #same for both
            ksa = ksb = att_kernel_size
        elif len(str(att_kernel_size))==2: #multiscale: first number for a, second number for b
            ksa = int(str(att_kernel_size)[0])
            ksb = int(str(att_kernel_size)[1])
        else: raise ValueError('invalid kernel_size specifier %s' % str(att_kernel_size))

        self.att_m = create_conv_bn_act(in_dim, hidden_dim, kernel_size=ksa, sep_conv=sep_conv,
                                        act=main_act if att_depth==1 else 'relu',
                                        dropout=dropout, dropout_type=dropout_type, groups=groups,
                                        sequential=False, padding='same', **kwargs)
        for i in range(att_depth-1):
            self.att_m.extend(create_conv_bn_act(hidden_dim, hidden_dim, kernel_size=ksa, sep_conv=sep_conv, act=main_act,
                                                 dropout=dropout, dropout_type=dropout_type, groups=groups,
                                                 sequential=False, padding='same', **kwargs))
        self.att_m = nn.Sequential(*self.att_m)

        self.att_gate = create_conv_bn_act(in_dim, hidden_dim, kernel_size=ksb, sep_conv=sep_conv,
                                           act=gate_act if att_depth==1 else 'relu',
                                           dropout=dropout, dropout_type=dropout_type, groups=groups,
                                           sequential=False, padding='same', **kwargs)
        for i in range(att_depth-1):
            self.att_gate.extend(create_conv_bn_act(hidden_dim, hidden_dim, kernel_size=ksa, sep_conv=sep_conv, act=gate_act,
                                                    dropout=dropout, dropout_type=dropout_type, groups=groups,
                                                    sequential=False, padding='same', **kwargs))
        self.att_gate = nn.Sequential(*self.att_gate)

        if dropout:
            self.short_name+= self.att_gate[-1].short_name

        for ign in ['norm', 'ws', 'kernel_size', 'padding']:
            kwargs.pop(ign,'')
        attention_c = create_conv_bn_act(hidden_dim, out_dim, kernel_size=1, act=att_act, norm=False, **kwargs)
        if len(attention_c)==1:
            self.att_last = attention_c[0]
        else:
            self.att_last = nn.Sequential(*attention_c)

        if in_att:
            self.short_name+='_'+in_att
        if groups>1:
            self.short_name+='_ab%d' % groups
        if main_act not in ['tanh']:
            self.short_name+='_ma'+main_act
        if gate_act not in ['sigm', 'sigmoid']:
            self.short_name+='_ga'+gate_act
        if att_kernel_size!=1:
            self.short_name+='_attks%s' % str(att_kernel_size)
            if sep_conv:
                self.short_name+='sc'
        if att_depth!=1:
            self.short_name+='_attd%d' % att_depth
        if att_act is not None:
            # self.att_act = create_activation(att_act)
            self.short_name+='_'+att_act

        if self.nms:
            self.short_name+='_nms%d' % self.nms
            if nmsinf: self.short_name+='i'


    def forward(self, x):
        a = self.att_m(x) #a,hdim,w,h
        b = self.att_gate(x)
        A = a.mul(b)
        A = self.att_last(A)  # N x n_classes

        if self.nms and (self.training or self.nmsinf):
            nms_set(A, kernel_size=self.nms, val=A.min())
        return A


class AttNet(nn.Module):
    def __init__(self, in_dim, out_dim, act='relu', depth=0, mask_zeros=True, topk=None, norm=False,
                 first_kernel_size=1, first_dropout=0, sep_conv=False, att_dropout=0, dropout_type='dropout',
                 head_depth=1, head_dropout=0, head_kernel_size=1, hidden_dim=None, in_att=None, pre_att=None,
                 att='gated', mb=False, mh=False, temperature=None,
                 inst_cluster=False, n_inst_cluster=8,
                 additive=False,
                 #groups=1,
                 **kwargs):
        """
        topk: take only the k highest attended values
        topk2: compute additionally the output when taking the topk2 highest attended values
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 512
        #     hidden_dim = 512 if in_dim >=512 else max(128, in_dim // 2)
        self.hidden_dim = hidden_dim
        self.mb = mb
        self.mh = mh

        self.additive = additive
        self.temperature = temperature

        self.mask_zeros = mask_zeros
        if mask_zeros:
            print('with masking zeros in att')

        # first = []
        # first.append(create_conv(in_dim, hidden_dim, kernel_size=1))
        # first.append(create_activation(act))
        # self.first = nn.Sequential(*first)

        self.first = create_conv_bn_act(in_dim, hidden_dim, kernel_size=first_kernel_size, padding='same',
                                        sep_conv=sep_conv, norm=norm, act=act, sequential=False,
                                        # groups=groups,
                                        att=in_att, dropout=first_dropout, dropout_type=dropout_type)
        if first_dropout:
            self.first.insert(0, create_dropout(first_dropout, dropout_type=dropout_type))
        self.first = nn.Sequential(*self.first)
        self.blocks = None
        blocks = []
        for d in range(depth):
            block = ResBlock(hidden_dim, hidden_dim, kernel_size=1, sep_conv=sep_conv, norm=norm, act=act,
                             #groups=groups
                             )
            blocks.append(block)
        if len(blocks)>0:
            self.blocks = nn.Sequential(*blocks)
        self.pre_att = create_attention(pre_att, hidden_dim)

        if att in ['gated',None]:
            ag_kwargs = dict(in_dim=hidden_dim, hidden_dim=hidden_dim//2, dropout_type=dropout_type,
                             norm=norm, out_dim=out_dim if mb else 1, **kwargs)
            if self.mh:
                self.att_net = ModuleConcat(AttGated2d, n=self.mh, dim=1, **ag_kwargs )
            else:
                self.att_net = AttGated2d(sep_conv=sep_conv, **ag_kwargs)
        else:
            raise ValueError('unknown attention %s' % att)
        if att_dropout:
            self.att_dropout = create_dropout(att_dropout, dropout_type=dropout_type, inplace=True)
        else:
            self.att_dropout = att_dropout
        # seq.append(att_net)
        # self.attention_net = nn.Sequential(*seq)
        # self.fc = create_conv(first_channels, n_classes, kernel_size=1)

        head_fct = self._create_head_additive if self.additive else self._create_head
        if mb:
            classifiers = [head_fct(hidden_dim, 1, depth=head_depth, act=act, norm=False, dropout=head_dropout, kernel_size=head_kernel_size)
                           for i in range(out_dim)]
            self.fc = nn.ModuleList(classifiers)
        else:
            self.fc = head_fct(hidden_dim, out_dim, depth=head_depth, act=act, norm=False, dropout=head_dropout, kernel_size=head_kernel_size)

        self.topk = topk

        #clams instance clustering
        self.inst_cluster = inst_cluster
        if inst_cluster:
            self.inst_classifiers = nn.ModuleList([create_linear(hidden_dim, out_dim) for i in range(out_dim)])
        self.n_inst_cluster = n_inst_cluster

        parts = []
        if in_att is not None:
            parts.append(in_att)
        if first_kernel_size!=1:
            parts.append('fks%d' % first_kernel_size)
        if sep_conv and (first_kernel_size!=1 or depth):
            parts.append('sc')
        if norm:
            if norm==True: norm='bn'
            parts.append('%s' % norm)
        if first_dropout:
            fdr_prefix = '2d' if dropout_type in ['drop2d', 'dropout2d'] else ''
            parts.append('ffdr%s%d' % (fdr_prefix, int(first_dropout*100)))
        if act!='relu':
            parts.append('%s' % act)
        if depth>0:
            parts.append('d%d' % depth)
        # if groups!=1:
        #     parts.append('gc%d' % groups)
        if head_depth>1:
            hpart = 'h%d' % head_depth
            if head_kernel_size>1:
                hpart+='ks%d'%head_kernel_size
            if head_dropout: hpart+= 'dr%d' % (head_dropout * 100)
            parts.append(hpart)
        elif head_kernel_size>1:
            parts.append('hks%d'%head_kernel_size)
        if self.pre_att is not None:
            parts.append(self.pre_att.short_name)
        parts.append(self.att_net.short_name)
        if att!='gated':
            parts.append(str(att))
        if self.temperature not in [None, 1]:
            parts.append('temp%s' % str(self.temperature))
        if mb: parts.append('mb')
        if mh: parts.append('mh%d' % mh)
        if hidden_dim!=512:
            parts.append('hdim%d' % hidden_dim)
        if att_dropout:
            if dropout_type in ['drop2d', 'dropout2d']:
                parts.append('adr2d%d' % (att_dropout*100))
            elif 'block' in dropout_type:
                parts.append('adrbl%d' % (att_dropout*100))
            else:
                parts.append('adr%d' % (att_dropout*100))
        if self.topk:
            parts.append('top%d' % topk)
            print('topk=%d' % topk)
        if self.inst_cluster:
            parts.append('instcluster')
        if self.additive:
            parts.append('add')
        name = '_'.join(parts)
        self.short_name = name

    def _create_head(self, in_dim, n_classes, depth=1, act='relu', norm=None, dropout=0, kernel_size=None):
        layers = []
        for hd in range(depth-1):
            if dropout:
                layers.append(create_dropout(dropout))
            hdim = max(in_dim//2, n_classes)
            layers.extend(create_linear_act(in_dim, hdim, norm=norm, sequential=False, act=act))
            in_dim = hdim
        layers.append(create_linear(in_dim, n_classes))
        if len(layers)==1:
            return layers[0]
        else:
            return nn.Sequential(*layers)

    def _create_head_additive(self, in_dim, n_classes, depth=1, act='relu', norm=False, dropout=0, pool=False,
                              kernel_size=1, sep_conv=True):
        layers = []
        for hd in range(depth-1):
            if dropout:
                layers.append(create_dropout(dropout))
            hdim = max(in_dim//2, n_classes)
            layers.extend(create_conv_bn_act(in_dim, hdim, kernel_size=kernel_size, norm=norm, sequential=False, act=act,
                                             padding='same', sep_conv=sep_conv))
            if pool: layers.append(create_pool())
            in_dim = hdim
        layers.append(create_conv(in_dim, n_classes, kernel_size=kernel_size, padding='same', sep_conv=sep_conv))
        if len(layers)==1:
            return layers[0]
        else:
            return nn.Sequential(*layers)

    def _determine_n_inst_cluster(self, A):
        return min(A.shape[-1]//2,self.n_inst_cluster)

    def _inst_out(self, A, h, inst_classifier):
        if h.shape[0]!=1:
            raise ValueError('implement support for batch_size>1')
        h = h.view(-1, h.shape[1]) #[1, 512, 128, 128] -> [16384, 512]
        A = A.view(1, -1) #[1, 1, 128, 128] -> [1, 16384]
        k_sample = self._determine_n_inst_cluster(A)
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids) #[8, 512])
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_logits = inst_classifier(top_p)[None, :, :]
        n_logits = inst_classifier(top_n)[None, :, :]
        return n_logits, p_logits

    def forward(self, x):
        #x e.g. [2, 512, 6, 8]
        x_in = x
        x = self.first(x)
        if self.blocks is not None:
            x = self.blocks(x)
        # h = x
        if self.pre_att:
            x = self.pre_att(x)

        result = {}

        A = self.att_net(x) #e.g. [2, 1, 6, 8] - [batch_dim, An, h, w]

        if self.inst_cluster:
            for i in range(len(self.inst_classifiers)):
                classifier = self.inst_classifiers[i]
                Ai = A[i] if self.mb else A
                neg_inst_logits, pos_inst_logits = self._inst_out(Ai, x, classifier)
                result[f'neg_inst_logits_{i}'] = neg_inst_logits
                result[f'pos_inst_logits_{i}'] = pos_inst_logits

        orig_shape = A.shape
        A = self._mask_att(A, x_in)
        A_raw = A
        A = self._dropout_att(A)

        A = A.view((A.shape[0],A.shape[1],-1)) # flatten/collapse h,w -> [batch, Adim, n_patches]
        A = filter_topk(A, self.topk)

        # if A.shape[0]!=1 and self.mb:
        #     print('check that this is correct when batch_size>1, A:', A.shape)

        # A = F.softmax(A, dim=-1)  # softmax over patches
        A = stable_softmax(A, dim=-1, temperature=self.temperature)
        A = A.view(orig_shape)

        if self.mh:
            A = A.sum(dim=1, keepdim=True)

        out = self.compute_out_A(x, A, result)


        # A = A.view(orig_shape) #return in orig shape before flattening
        result.update({'out':out, 'A':A_raw, 'A_soft':A#, 'm':m
                       })
        return result

    def compute_out_A(self, x, A, result):
        if self.additive:
            if self.mb:
                m = [A[:, i].unsqueeze(1) * x for i in range(A.shape[1])]
            else:
                m = A * x
            out, A_logits = self._compute_out_additive(m)
            result['A_out'] = A_logits
        else:
            A = A.view((A.shape[0], A.shape[1], -1)) #flatten w,h, e.g. [2, 1, 48]
            x = x.view((x.shape[0], x.shape[1], -1))  # flatten w,h, e.g. [2, 512, 48]
            x = torch.transpose(x, 2, 1)  # now [2, 48, 512]
            m = torch.bmm(A, x)  # e.g. [2, 1, 48] x [2, 48, 512] -> [2, 1, 512]

            out = self._compute_out(A, m)
        return out

    def _compute_out(self, A, m):
        if A.shape[1] > 1:  # mb, A.shape[1]=out_dim
            outs = []
            for c in range(A.shape[1]):
                mc = m[:, [c]]  # [b,1,features], e.g [2,1,512]
                outc = self.fc[c](mc.squeeze(-2))
                outs.append(outc)
            out = torch.hstack(outs)
        else:
            m = m.squeeze(dim=1)  # squeezed to [2, 512]
            out = self.fc(m)
        if m.isnan().any():
            print('x has nans!')

        return out

    def _compute_out_additive(self, m):
        if self.mb:
                patch_logits = []
                for c,mc in enumerate(m):
                    if mc.isnan().any():
                        print('m%d nans!' % c)
                    patch_logits_c = self.fc[c](mc)
                    patch_logits.append(patch_logits_c)
                patch_logits = torch.hstack(patch_logits)
        else:
            # m = m.squeeze(dim=1)
            patch_logits = self.fc(m)
            if m.isnan().any():
                print('m has nans!')

        out = torch.sum(patch_logits, dim=[-1,-2], keepdim=False)

        return out, patch_logits

    def _mask_att(self, A, x_in):
        #A: [batch, Adim, h, w]
        # Amin = A.min().detach()
        if self.mask_zeros:  # todo works only if first block doesnt change size
            mask = torch.std(x_in, 1).detach() == 0
            mask = mask.unsqueeze(1).repeat(1, A.shape[1], 1, 1)
            A = A.masked_fill(mask, float('-inf'))
        # A = A_raw  # [batch, Adim, h, w]
        return A

    def _dropout_att(self, A):
        if self.att_dropout and self.training:
            # mask_ = torch.rand(A.shape, device=A.device)<=self.att_dropout.p
            mask = torch.ones_like(A, device=A.device)
            mask = self.att_dropout(mask)
            mask = mask == 0
            A = A.masked_fill(mask, float('-inf'))
            # A = A.masked_fill(mask, Amin)
            # print('mask==0:',(mask==0).sum().item(), 'mask==1:',(mask==1).sum().item())
        return A


def filter_topk(A, topk):
    assert len(A.shape)==3 #batch,Adim,p
    n_el = A.shape[-1]
    if topk and n_el>topk:
        mask = torch.zeros_like(A)
        mask.fill_(float('-inf'))
        smallest, indices = torch.topk(A, k=n_el-topk, largest=False, dim=-1)
        A = A.scatter(-1, indices, mask)
    return A


def _example(**kwargs):
    # anet = AttNet(1024, 2, first_kernel_size=3, sep_conv=True)
    # anet = AttNet(1024, 2, depth=1, att_dropout=0.5)
    anet = AttNet(1024, 2, hidden_dim=256, norm=True, head_depth=0, mb=False, additive=True, **kwargs)
    anet.training = True
    inp = torch.zeros((10, 1024, 6, 8))
    print_model_summary(anet, inp)
    out = anet(inp)
    print(inp.shape, out['out'].shape)
    A = out['A']
    print(A.shape, A.min(), A.max())

#Reimplementation of CLAMs instance clustering loss to be able to correctly compare to CLAM. Doesnt seem to have
#much effect in practice.
class InstClusterLoss(object):
    """ todo: fkt. so nicht """
    def __init__(self, n_classes, subtyping, k=8):
        self.n_classes = n_classes
        self.k = k
        self.subtyping = subtyping
        self.criterion = nn.CrossEntropyLoss()

    def set_module(self, modul:ModuleBase):
        # self.metrics['train_topk'] = {'acc':torchmetrics.Accuracy(num_classes=n_train)}
        self.modul = modul

    def _create_targets(self, n, device, positive=False):
        return torch.ones((n), dtype=torch.long, device=device)*int(positive)

    def __call__(self, batch, batch_idx, loss, result, x, y):
        self._batch_size = len(x)
        assert self._batch_size == 1  # assumes single batch for simplicity

        if len(y.size())==1:
            inst_labels = F.one_hot(y, num_classes=self.n_classes).squeeze() #binarize label
        else:#multilabel already hot encoded
            inst_labels = y.squeeze()
        assert len(inst_labels)==self.n_classes
        assert inst_labels.max().item()==1

        inst_loss = 0
        for i in range(self.n_classes):
            inst_label = inst_labels[i].item()
            if inst_label == 0 and not self.subtyping:
                continue

            neg_logits = result[f'neg_inst_logits_{i}'].squeeze()
            pos_logits = result[f'pos_inst_logits_{i}'].squeeze()
            neg_targets = self._create_targets(neg_logits.shape[0], x.device, positive=False)
            pos_targets = self._create_targets(pos_logits.shape[0], x.device, positive=True)
            logits = torch.cat((neg_logits, pos_logits), dim=0)
            targets = torch.cat((neg_targets, pos_targets), dim=0)
            inst_loss = inst_loss + self.criterion(logits, targets)
        inst_loss /= self.n_classes
        loss = loss + inst_loss
        self.modul.log("train_inst_loss_step", inst_loss, batch_size=self._batch_size)
        self.modul._aggregate_loss(inst_loss, batch_idx, phase='train_inst')
        self.modul._compute_metrics_step(target=targets, logits=logits, phase='train_inst')
        return loss

    def training_epoch_end(self):
        self.modul._log_loss_epoch(phase='train_inst')
        self.modul._compute_metrics_epoch(phase='train_inst')

if __name__ == '__main__':
    _example()


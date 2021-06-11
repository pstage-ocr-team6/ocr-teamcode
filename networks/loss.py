import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    
    def __init__(self, eps: float = 0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps, self.reduction = eps, reduction
        self.ignore_index = ignore_index

    def forward(self, output, target, *args):
        #print('output shape', output.shape)
        #print('target shape', target.shape)
        output = output.transpose(1,2)
        
        pred = output.contiguous().view(-1, output.shape[-1])
        target = target.to(pred.device).contiguous().view(-1)
        c = pred.size()[-1]
        #print('pred shape', pred.shape)
        #print('target shape', target.shape)
        #print('c shape', c)

        log_preds = F.log_softmax(pred, dim=-1)
        # print(">> pred의 크기", pred.size())

        # ignore index for smooth label > 타겟 중에 패딩 아닌곳 위치
        ignore_target = target != self.ignore_index
        # log pred 12250, ignore_index_target 3100
        log_preds = log_preds * ignore_target[:, None]
        # print(">> target:", target)
        # print(">> log_preds:", log_preds)
        # print(">> log_preds의 크기:", log_preds.size())

        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * \
               F.nll_loss(log_preds, target, reduction=self.reduction,
                          ignore_index=self.ignore_index)
        
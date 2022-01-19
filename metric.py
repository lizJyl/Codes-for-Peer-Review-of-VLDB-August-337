import numpy as np



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def iou_loss(output, label):
    output = output.view(output.shape[0], -1)
    label = label.view(label.shape[0], -1)
    intsec = torch.min(output, label).sum(1)
    union = torch.max(output, label).sum(1)
    one_vec = torch.ones(output.shape[0]).float()
    loss = one_vec - intsec/union
    return loss.mean()

def iou_loss_merge(output, label):
    output = output.view(-1)
    label = label.view(-1)
    intsec = torch.min(output, label).sum()
    union = torch.max(output, label).sum()
    loss = 1. - intsec*1./union
    return loss

def f1_loss(output, label):
    output = output.view(output.shape[0], -1)
    label = label.view(label.shape[0], -1)
    iou_thr=0.5
    tp = ((output > iou_thr) * (label > iou_thr)).sum()
    fp = ((output > iou_thr) * (label < iou_thr)).sum()
    fn = ((output < iou_thr) * (label > iou_thr)).sum()
    # if tp + fp > 0:
    precision = tp * 1. / (tp + fp+1E-9)
    # else:
    #   precision = 0
    # if tp + fn > 0:
    recall = tp * 1. / (tp + fn+1E-9)
    # else:
    #   recall = 0
    # if precision + recall > 0:
    f1_score = 2 * precision * recall / (precision + recall+1E-9)
    # else:
    #   f1_score = 0
    one_vec = torch.ones(output.shape[0]).float()
    loss=one_vec- f1_score
    return loss.mean()

def calc_f1_score(output, label):
    output = np.array(output, dtype=np.float)
    label = np.array(label, dtype=np.float)
    # tp = ((output+label)==2).sum()
    # fp = ((output-label)==1).sum()
    # fn = ((label-output)==1).sum()
    tp = ((output + label) >1.5 ).sum()
    fp = ((output - label) >0.5 ).sum()
    fn = ((label - output) >0.5 ).sum()
    print("tp, fp, fn ",tp,fp,fn)

    if tp+fp > 0:
        precision = tp*1./(tp+fp)
    else:
        precision = 0
    if tp+fn > 0:
        recall = tp*1./(tp+fn)
    else:
        recall = 0
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    print("precision",precision)
    print("recall",recall)
    return precision, recall, f1_score

    # print("precision",precision)
    # print("recall",recall)
    #
    #
    # if f1_score > best_score[2]:
    #   best_score[0] = precision
    #   best_score[1] = recall
    #   best_score[2] = f1_score
    #   return f1_score, True
    # else:
    #   return f1_score, False

def calc_f1_score_iouthr(output, label, iou_thr=0.5):
    output = np.array(output, dtype=np.float)
    label = np.array(label, dtype=np.float)
    # tp = ((output+label)==2).sum()
    # fp = ((output-label)==1).sum()
    # fn = ((label-output)==1).sum()
    # tp = ((output + label) >1.5 ).sum()
    # fp = ((output - label) >0.5 ).sum()
    # fn = ((label - output) >0.5 ).sum()
    tp = ((output > iou_thr) * (label > iou_thr)).sum()
    fp = ((output > iou_thr) * (label < iou_thr)).sum()
    fn = ((output < iou_thr) * (label > iou_thr)).sum()
    # print("tp, fp, fn ",tp,fp,fn)

    if tp+fp > 0:
        precision = tp*1./(tp+fp)
    else:
        precision = 0
    if tp+fn > 0:
        recall = tp*1./(tp+fn)
    else:
        recall = 0
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    # print("precision",precision)
    # print("recall",recall)
    return precision, recall, f1_score

import torch
import torch.nn.functional as F


def sampling(pred_means, pred_vars):
    """
    High std means low uncertainty. Therefore, divided by 1

    :param pred_means:
    :param pred_vars:
    :return:
    """
    # 5 models, each sampled 10 times = 50,
    sample1 = torch.distributions.Normal(pred_means[0], pred_vars[0]).sample(
        [10])
    sample2 = torch.distributions.Normal(pred_means[1], pred_vars[1]).sample(
        [10])
    sample3 = torch.distributions.Normal(pred_means[2], pred_vars[2]).sample(
        [10])
    sample4 = torch.distributions.Normal(pred_means[3], pred_vars[3]).sample(
        [10])
    sample5 = torch.distributions.Normal(pred_means[4], pred_vars[4]).sample(
        [10])
    samples = torch.cat((sample1, sample2, sample3, sample4, sample5))
    # Samples = [5 * 10, 10 predictions, 11 state dims]
    # print(samples.shape)
    stds = torch.var(samples, dim=0)
    # print(stds.shape)
    # [10 predictions, 11 state dims]
    total_stds = torch.mean(stds, dim=1)
    total_stds = F.sigmoid(total_stds)
    # total_stds = 1 / total_stds
    # total_stds = total_stds / torch.mean(total_stds)  # if very uncertain,
    # high std, encouraged.
    # total_stds = total_stds - torch.min(total_stds)
    return total_stds.detach()
import torch


def vi(mean, var):
    # Distance to unit gaussian, means how certain it is high distance means high certainty.
    # Loss is small: uncertain. Loss is high: certain.
    multi_p = torch.distributions.normal.Normal(mean, var)
    multi_q = torch.distributions.normal.Normal(
        torch.zeros(mean.shape), torch.ones(var.shape)
    )
    multi_loss = torch.distributions.kl_divergence(multi_p, multi_q)
    multi_loss = torch.sum(multi_loss, dim=0)
    multi_loss = torch.sum(multi_loss, dim=1)
    if mean.shape[1] == 1:
        print("--------------------")
        return multi_loss.item()
    # if multi_loss.shape
    min = torch.min(multi_loss)
    max = torch.max(multi_loss)
    scale = (max - min)
    scale[torch.abs(scale) < 0.001] = 0.001
    # [0 - 1]: Certainty.
    multi_loss = (multi_loss - min) / scale
    # Uncertainty.
    multi_loss = 1.0 - multi_loss
    return multi_loss.unsqueeze(dim=1).detach()


def mean_std(mean, var):
    total_stds = torch.std(mean, dim=0)
    total_stds = torch.sum(total_stds, dim=1)
    return total_stds.detach().item()


def sampling(pred_means, pred_vars):
    # 5 models, each sampled 10 times = 50,
    sample1 = torch.distributions.Normal(pred_means[0], pred_vars[0]).sample([10])
    sample2 = torch.distributions.Normal(pred_means[1], pred_vars[1]).sample([10])
    sample3 = torch.distributions.Normal(pred_means[2], pred_vars[2]).sample([10])
    sample4 = torch.distributions.Normal(pred_means[3], pred_vars[3]).sample([10])
    sample5 = torch.distributions.Normal(pred_means[4], pred_vars[4]).sample([10])
    samples = torch.cat((sample1, sample2, sample3, sample4, sample5), dim=0)
    # Samples = [5 * 10, 10 predictions, 11 state dims]
    stds = torch.std(samples, dim=0)
    # [10 predictions, 11 state dims]
    total_stds = torch.mean(stds, dim=1)
    # total_stds = total_stds
    # total_stds = total_stds / torch.mean(total_stds)  # if very uncertain, high std, encouraged.
    # total_stds = total_stds - torch.min(total_stds)
    return total_stds.item()

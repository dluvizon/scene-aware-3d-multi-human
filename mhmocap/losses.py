import torch

from .transforms import softplus


def build_eigen_loss_fn(eps=1e-3):
    
    def _depth_loss_fn(y_pred, y_true, mask):
        d_pred = mask * torch.log(torch.clamp(y_pred, eps))
        d_true = mask * torch.log(torch.clamp(y_true, eps))
        N = torch.sum(mask) + 1.0
        diff_log = torch.sum(torch.square(d_pred - d_true))

        return diff_log / N
        
    return _depth_loss_fn


def build_avg_depth_loss_fn(eps=1e-3):
    
    def _depth_loss_fn(y_pred, y_true, mask):
        d_pred = mask * torch.log(torch.clamp(y_pred, eps))
        d_true = mask * torch.log(torch.clamp(y_true, eps))
        mask_tn = torch.sum(mask, dim=(2, 3))
        d_pred_tn = torch.sum(d_pred, dim=(2, 3)) / (mask_tn + 1)
        d_true_tn = torch.sum(d_true, dim=(2, 3)) / (mask_tn + 1)
        diff_log = torch.sum(torch.square(d_pred_tn - d_true_tn))
        return diff_log
        
    return _depth_loss_fn


def build_masked_mse_loss_fn():

    def _masked_mse_loss_fn(y1, y2, mask):
        N = torch.sum(mask) + 1.0
        diff = torch.sum(torch.square(mask * (y1 - y2)))
        return diff / N
        
    return _masked_mse_loss_fn


def inv_depth_loss_fn(diff_z):
    return torch.pow(softplus(diff_z), 2.0)


def build_multiscale_grad_loss(num_grad_levels=4, alpha=0.5):
    """Build a function to compute a gradient based loss (L1 + grad).
    This loss function is specially suitable to enforce sharp borders.
    The arguments of the loss function should be:
        y_true: tensor with shape (H, W, ...)
        y_pred: tensor with shape (H, W, ...)
        y_mask: tensor with shape (H, W, ...)
    """
    depth_loss = build_eigen_loss_fn(lamb=0.0)

    def _recursive_multiscale_grad_loss(diff, mask, num_grad_levels):
        v_grad = torch.abs(diff[:-1, ...] - diff[1:, ...])
        v_mask = mask[:-1, ...] * mask[1:, ...]
        v_N = torch.clamp(torch.sum(v_mask, dim=(0, 1)), 1, None)
        v_grad = torch.sum(v_mask * v_grad, dim=(0, 1)) / v_N

        h_grad = torch.abs(diff[:, :-1] - diff[:, 1:])
        h_mask = mask[:, :-1] * mask[:, 1:]
        h_N = torch.clamp(torch.sum(h_mask, dim=(0, 1)), 1, None)
        h_grad = torch.sum(h_mask * h_grad, dim=(0, 1)) / h_N

        loss = torch.mean(v_grad + h_grad)

        min_size = min(diff.shape[-2:])
        if (num_grad_levels > 1) and (min_size >= 4):
            diff = torch.cat((
                diff[0::2, 0::2, ...],
                diff[0::2, 1::2, ...],
                diff[1::2, 0::2, ...],
                diff[1::2, 1::2, ...],
            ), dim=0)
            mask = torch.cat((
                mask[0::2, 0::2, ...],
                mask[0::2, 1::2, ...],
                mask[1::2, 0::2, ...],
                mask[1::2, 1::2, ...],
            ), dim=0)

            loss += _recursive_multiscale_grad_loss(diff, mask, num_grad_levels=num_grad_levels - 1)

        return loss

    def _multiscale_grad_loss(y_true, y_pred, mask):
        loss = depth_loss(y_true, y_pred, mask)

        if num_grad_levels > 0:
            #diff = y_true - y_pred
            diff = torch.log(torch.clamp(y_true, 1e-3)) - torch.log(torch.clamp(y_pred, 1e-3))
            if len(y_true.shape) < 3:
                diff = diff.unsqueeze(-1)
                mask = mask.unsqueeze(-1)

            loss += alpha * _recursive_multiscale_grad_loss(diff, mask, num_grad_levels - 1)

        return loss

    return _multiscale_grad_loss

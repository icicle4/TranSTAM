import math
import sys
from typing import Iterable
import pdb

import torch
from utils import misc as utils


def to_cuda(samples, device):
    if isinstance(samples, torch.Tensor):
        samples = samples.to(device, non_blocking=True)
    elif isinstance(samples, list):
        samples = [to_cuda(item, device) for item in samples]
    elif isinstance(samples, dict):
        samples = {key: to_cuda(samples[key], device) for key in samples.keys()}
    return samples


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    data_loader = iter(data_loader)
    with torch.autograd.set_detect_anomaly(True):
        for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
            samples = next(data_loader)
            outputs = model(samples, stage="train")
            loss_dict, accuracy_dict = criterion(outputs, samples["labels"])
            losses = sum(loss_dict[k] for k in loss_dict.keys())

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            accuracy_dict_redued = utils.reduce_dict(accuracy_dict)
            loss_dict_reduced = {
                f'{k}': v.item()
                for k, v in loss_dict_reduced.items()
            }
            accuracy_dict_redued = {
                f'{k}': v.item()
                for k, v in accuracy_dict_redued.items()
            }

            losses_reduced = sum(loss_dict_reduced.values())
            loss_value = losses_reduced

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()

            losses.backward()

            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(**accuracy_dict_redued)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    to_log_metric_logger = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return to_log_metric_logger


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, epoch: int):

    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Test Epoch: [{}]'.format(epoch)
    print_freq = 10

    data_loader = iter(data_loader)

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        samples = next(data_loader)

        outputs = model(samples, stage="test")
        loss_dict, accuracy_dict = criterion(outputs, samples["labels"])

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        accuracy_dict_redued = utils.reduce_dict(accuracy_dict)
        loss_dict_reduced = {
            f'{k}': v.item()
            for k, v in loss_dict_reduced.items()
        }
        accuracy_dict_redued = {
            f'{k}': v.item()
            for k, v in accuracy_dict_redued.items()
        }

        losses_reduced = sum(loss_dict_reduced.values())
        loss_value = losses_reduced

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(**accuracy_dict_redued)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    to_log_metric_logger = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return to_log_metric_logger


import os
import numpy as np
from configargparse import ArgumentParser
from typing import Any
from copy import deepcopy
import math
from tqdm import tqdm
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim import Adam, SGD

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import network as models

from optimiser import LARSSGD, collect_params

class BYOL(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 learning_rate: float = 0.2,
                 weight_decay: float = 1.5e-6,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 warmup_epochs: int = 0,
                 max_epochs: int = 1,
                 o_units: int = 256,
                 h_units: int = 4096,
                 model: str = 'resnet18',
                 tau: float = 0.996,
                 optimiser: str = 'sgd',
                 effective_bsz: int = 256,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.norm_layer == 'GN':

            self.hparams.norm_l = nn.GroupNorm
        else:
            self.hparams.norm_l = nn.BatchNorm2d

        self.encoder_online = getattr(models, self.hparams.model)(
            dataset=self.hparams.dataset, norm_layer=self.hparams.norm_l)

        self.encoder_online.fc = Identity()

        self.proj_head_online = models.projection_MLP(model=self.hparams.model,
                                                      output_dim=self.hparams.o_units, hidden_dim=self.hparams.h_units, norm_layer=self.hparams.norm_l)

        self.predictor_theta_online = models.projection_pred(
            input_dim=self.hparams.o_units, output_dim=self.hparams.o_units, hidden_dim=self.hparams.h_units, norm_layer=self.hparams.norm_l)
    
        self.encoder_target = deepcopy(self.encoder_online)
        self.proj_head_target = deepcopy(self.proj_head_online)
        
        for param_t in self.encoder_target.parameters():
            param_t.requires_grad = False
        
        for param_t in self.proj_head_target.parameters():
            param_t.requires_grad = False

        self.initial_tau = self.hparams.tau
        self.current_tau = self.hparams.tau

        # self.init_trainlog()
        self.effective_bsz = effective_bsz
        
        print("\n\n\n effective_bsz:{} \n\n\n".format(self.effective_bsz))

        self.train_loss = []
        self.valid_loss = []

    def init_trainlog(self):
        print("log_path: {}".format(self.hparams.log_path))
        os.makedirs(self.hparams.log_path, exist_ok=True)

        # reset root logger
        [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
        # info logger for saving command line outputs during training
        logging.basicConfig(level=logging.INFO, format='%(message)s',
                            handlers=[logging.FileHandler(os.path.join(self.hparams.log_path, 'trainlogs.txt')),
                                      logging.StreamHandler()])

    def shared_step(self, batch, batch_idx):

        # update tau after
        self.current_tau = self.update_tau()

        (img_i, img_j), y = batch

        z_i = self.encoder_online(img_i)
        z_j = self.encoder_online(img_j)

        o_z_i = self.proj_head_online(z_i)
        o_z_j = self.proj_head_online(z_j)
        
        p_i = self.predictor_theta_online(o_z_i)
        p_j = self.predictor_theta_online(o_z_j)

        with torch.no_grad():
            # update weights
            self.update_weights()

            t_z_i = self.encoder_target(img_i)
            t_z_j = self.encoder_target(img_j)

            o_t_z_i = self.proj_head_target(t_z_i)
            o_t_z_j = self.proj_head_target(t_z_j)

        loss_1_2 = self.compute_loss(p_i, o_t_z_j.detach())
        loss_2_1 = self.compute_loss(p_j, o_t_z_i.detach())

        loss = (loss_1_2 + loss_2_1).mean()

        # Testing Printing loss functions
        # print("loss :{}".format((loss_1_2 + loss_2_1).mean()))
        # print("i_on_loss :{}".format(i_on_loss))
        # print("j_on_loss :{}".format(j_on_loss))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'loss': loss}, prog_bar=True, on_epoch=True)
        self.train_loss.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'val_loss': loss}, prog_bar=True, on_epoch=True)

        self.valid_loss.append(loss.item())

        # return loss

    def configure_optimizers(self):

        lr = (self.hparams.learning_rate * (self.effective_bsz / 256))

        params = list(self.encoder_online.parameters()) + \
            list(self.predictor_theta_online.parameters()) + \
            list(self.proj_head_online.parameters())

        if self.hparams.optimiser == 'lars':

            models = [self.encoder_online, self.predictor_theta_online, self.proj_head_online]

            param_list = collect_params(models, exclude_bias_and_bn=True)

            # print(params)

            optimizer = LARSSGD(
                param_list, lr=lr, weight_decay=self.hparams.weight_decay, eta=0.001, nesterov=False)

        elif self.hparams.optimiser == 'adam':
            optimizer = Adam(params, lr=lr,
                             weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimiser == 'sgd':
            optimizer = SGD(params, lr=lr,
                            weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
        else:
            raise NotImplementedError('{} not setup.'.format(self.ft_optimiser))

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=1e-3 * lr
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        (args, _) = parser.parse_known_args()

        # Data
        parser.add_argument('--dataset', type=str, default='cifar10',
                            help='cifar10, imagenet')
        parser.add_argument('--data_dir', type=str, default=None)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--jitter_d', type=float, default=0.5)
        parser.add_argument('--jitter_p', type=float, default=0.8)
        parser.add_argument('--blur_p', type=float, default=0.5)
        parser.add_argument('--grey_p', type=float, default=0.2)
        parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0])

        # optim
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=0.02)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--warmup_epochs', type=float, default=10)
        parser.add_argument('--optimiser', default='sgd',
                            help='Optimiser, (Options: sgd, adam, lars).')

        # Model
        parser.add_argument('--model', default='resnet18',
                            help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
        parser.add_argument('--h_units', type=int, default=4096)
        parser.add_argument('--o_units', type=int, default=256)
        parser.add_argument('--tau', type=float, default=0.996)
        parser.add_argument('--norm_layer', default=nn.BatchNorm2d)

        parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true',
                            help='Save the checkpoints to Neptune (Default: False)')
        parser.set_defaults(save_checkpoint=False)
        parser.add_argument('--print_freq', type=int, default=1)

        return parser

    def compute_loss(self, x, y):

        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)

        loss = 2 - 2 * (x * y).sum(dim=-1)

        return loss

    def update_tau(self):
        max_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * \
            (math.cos(math.pi * (self.global_step+1) / max_steps) + 1) / 2
        return tau

    def update_weights(self):
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(self.encoder_online.named_parameters(), self.encoder_target.named_parameters()):
            
            target_p.data = (target_p.data * self.current_tau) + \
                (online_p.data * (1. - self.current_tau))

        for (name, online_p), (_, target_p) in zip(self.proj_head_online.named_parameters(), self.proj_head_target.named_parameters()):
            
            target_p.data = (target_p.data * self.current_tau) + \
                (online_p.data * (1. - self.current_tau))


class Identity(torch.nn.Module):
    """
    An identity class to replace arbitrary layers in pretrained models
    Example::
        from pl_bolts.utils import Identity
        model = resnet18()
        model.fc = Identity()
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

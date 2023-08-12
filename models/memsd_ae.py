from torchmetrics import AUROC
import torch
from torch import nn
import lightning as L
from losses import EntropyLossEncap
import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
import torch.nn.functional as F
from utils import get_radius
from losses import SSIM
from losses import ssim
from ..mem import MemModule
from ..ae import Encoder, Decoder
# from losses import EntropyLossEncap
# import sys
# import os
# sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(__file__))
# from models.cifar10 import MemAEV1
from .mem_ae import MemAEV


class CIFAR10MemAESDV1(MemAEV):
    def __init__(self,
                 chnum_in,
                 seed,
                 objective,
                 center=None,
                 nu: float = 0.1,
                 entropy_loss_weight=0.0002,
                 shrink_thres=0.0025,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestone=[250],
                 optimizer_name='amsgrad',
                 log_red=False):
        super().__init__(chnum_in, seed, entropy_loss_weight, shrink_thres,
                         rep_dim, lr, weight_decay, lr_milestone,
                         optimizer_name, log_red)
        self.lr = lr
        self.objective = objective
        self.weight_decay = weight_decay
        self.lr_milestone = lr_milestone
        self.entropy_loss_weight = entropy_loss_weight
        self.optimizer_name = optimizer_name
        self.mse = nn.MSELoss(reduction='mean')
        self.log_red = log_red
        self.rep_dim = rep_dim
        self.center = torch.zeros(rep_dim) if center is None else center
        self.objective = objective
        self.warm_up_n_epochs = 10
        self.R = torch.tensor(0)
        self.nu = nu

        self.validation_step_outputs = []

    def init_center_c(self, net, train_loader, eps=0.1):
        n_samples = 0
        c = torch.zeros(1024).cuda()
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.cuda()
                outputs = net(inputs)['mem_out']
                outputs = outputs.contiguous().view(outputs.size(0), -1)

                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def forward(self, x):
        enc_x = self.encoder(x)
        res_mem = self.mem_rep(enc_x)
        mem_x = res_mem['output']
        att = res_mem['att']
        dec_x = self.decoder(mem_x)
        return {
            'enc_out': enc_x,
            'dec_out': dec_x,
            'att': att,
            'mem_out': mem_x
        }

    def on_validation_epoch_start(self) -> None:
        self.val_idx_label_score = []

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)
        mem_out = outputs['mem_out']
        dec_out = outputs["dec_out"]
        mem_out = mem_out.contiguous().view(mem_out.size(0), -1)
        dist = torch.sum((mem_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
        else:
            scores = dist
        # loss = torch.mean(scores)
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        svdd_mse_scores = scores + mse_scores
        self.validation_step_outputs += zip(labels, scores, mse_scores,
                                            svdd_mse_scores)

    def on_validation_epoch_end(self):
        auroc = AUROC(task="binary")
        labels, scores, mse_scores, svdd_mse_scores = zip(
            *self.validation_step_outputs)
        auroc_score = auroc(torch.stack(scores), torch.stack(labels))
        auroc_mse_score = auroc(torch.stack(mse_scores), torch.stack(labels))
        auroc_svdd_mse_score = auroc(torch.stack(svdd_mse_scores),
                                     torch.stack(labels))
        # print(auroc_score)
        self.log('val_roc_auc', auroc_score, prog_bar=True, sync_dist=True)
        self.log('mse_roc_auc', auroc_mse_score, prog_bar=True, sync_dist=True)
        self.log('svdd_mse_roc_auc',
                 auroc_svdd_mse_score,
                 prog_bar=True,
                 sync_dist=True)
        self.validation_step_outputs.clear()

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs)
        att_out = outputs["att"]
        mem_out = outputs['mem_out']
        dec_out = outputs["dec_out"]
        mem_out = mem_out.contiguous().view(mem_out.size(0), -1)
        mse_loss = self.mse(inputs, dec_out)
        dist = torch.sum((mem_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch >=
                                                    self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)

        entropy_loss = self.entropy_loss_func(att_out)
        loss = svdd_loss + self.svdd_loss_weight * mse_loss + self.entropy_loss_weight * entropy_loss
        self.log("train_loss", loss, sync_dist=True)
        # self.log_tsne(outputs["dec_out"], self.current_epoch)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.optimizer_name == 'amsgrad')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestone, gamma=0.1)
        # return optimizer, scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

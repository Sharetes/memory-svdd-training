from torch import nn
from functools import partial
import pytorch_lightning as pl
from losses import EntropyLossEncap
from lightning.fabric import seed_everything
import torch
from torchmetrics import AUROC

from losses import SSIM
from losses import ssim
# import sys
# import os
# sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(__file__))

# from models.cifar10 import MemModule
from .mem import MemModule


class Encoder(nn.Module):
    def __init__(self, chnum_in, feature_num, feature_num_2,
                 feature_num_x2) -> None:
        super().__init__()
        m_conv2d = partial(nn.Conv2d, kernel_size=3, stride=2, padding=1)
        self.encoder = nn.Sequential(
            m_conv2d(chnum_in, feature_num_2), nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2,
                      feature_num,
                      bias=False,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num,
                      feature_num_x2,
                      bias=False,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2,
                      feature_num_x2,
                      bias=False,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input_x):
        return self.encoder(input_x)


class Decoder(nn.Module):
    def __init__(self, chnum_in, feature_num, feature_num_2,
                 feature_num_x2) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_num_x2,
                               feature_num_x2,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(feature_num_x2), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_x2,
                               feature_num,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1), nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num,
                               feature_num_2,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(feature_num_2), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_2,
                               chnum_in,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1))

    def forward(self, input_x):
        return self.decoder(input_x)


class MemAEV(pl.LightningModule):
    def __init__(
        self,
        chnum_in,
        seed,
        entropy_loss_weight=0.0002,
        shrink_thres=0.0025,
        rep_dim=128,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestone=[250],
        optimizer_name='amsgrad',
        log_red=False,
    ):
        super().__init__()
        self.chnum_in = chnum_in
        self.save_hyperparameters()
        # print(self.hparams)
        seed_everything(seed)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestone = lr_milestone
        self.optimizer_name = optimizer_name
        self.entropy_loss_weight = entropy_loss_weight
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = Encoder(chnum_in, feature_num, feature_num_2,
                               feature_num_x2)

        self.decoder = Decoder(chnum_in, feature_num, feature_num_2,
                               feature_num_x2)

        self.mem_rep = MemModule(mem_dim=rep_dim,
                                 fea_dim=feature_num_x2,
                                 shrink_thres=shrink_thres)
        self.entropy_loss_func = EntropyLossEncap().cuda()
        self.mse = nn.MSELoss(reduction='mean')
        self.validation_step_outputs = []

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'dec_out': output, 'att': att, 'mem_out': f}

   def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs)
        att_out = outputs["att"]
        mse_loss = self.mse(inputs, outputs["dec_out"])
        entropy_loss = self.entropy_loss_func(att_out)
        loss = mse_loss
        self.log("train_loss", loss)
        # self.log_tsne(outputs["dec_out"], self.current_epoch)
        return {'loss': loss, "dec_out": outputs["dec_out"], "labels": labels} 

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)["dec_out"]
        scores = torch.sum((outputs - inputs)**2,
                           dim=tuple(range(1, outputs.dim())))
        # loss = torch.mean(scores)
        self.validation_step_outputs += zip(labels, scores)

    def on_validation_epoch_end(self):
        auroc = AUROC(task="binary")
        labels, scores = zip(*self.validation_step_outputs)
        auroc_score = auroc(torch.stack(scores), torch.stack(labels))
        # print(auroc_score)
        self.log('val_roc_auc', auroc_score, prog_bar=True, sync_dist=True)
        # self.logger.log_hyperparams(self.hparams, {
        #     "auc_roc": auroc_score,
        # })
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.optimizer_name == 'amsgrad')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestone, gamma=0.1)
        # return optimizer, scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


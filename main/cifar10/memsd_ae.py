import os
import sys
# from sklearn.manifold import TSNE

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from models import CIFAR10MemAESDV
from models import MemAEV
from dataset import CIFAR10DataModel
# from utils import display_tsne
# from utils import TSNE
from utils import transfer_weights
from utils import load_pre_ae_model
from utils import init_envir

# def transfer_weights(dst_net, src_net):
#     """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

#     dst_net_dict = dst_net.state_dict()
#     src_net_dict = src_net.state_dict()
#     # Filter out decoder network keys
#     src_net_dict = {
#         k: v
#         for k, v in src_net_dict.items()
#         if k in dst_net_dict and not k.startswith("fc1")
#     }
#     # print(src_net_dict.keys())
#     dst_net_dict.update(src_net_dict)
#     # Load the new state_dict
#     dst_net.load_state_dict(dst_net_dict)


def cifar10_lenet(bash_log_name,
                  normal_class,
                  seed,
                  pre_epochs,
                  epochs,
                  log_path,
                  objective,
                  radio,
                  batch_size,
                  devices=2,
                  enable_progress_bar=False):
    # log_path = log_path + datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')[:-3]
    cifar10 = CIFAR10DataModel(batch_size=batch_size,
                               radio=radio,
                               normal_class=normal_class)
    auto_enc = MemAEV(chnum_in=3, seed=seed)
    # auto_enc.encoder = torch.compile(auto_enc.encoder)
    # auto_enc.decoder = torch.compile(auto_enc.decoder)
    # auto_enc = torch.compile(auto_enc, mode="reduce-overhead")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        # enable_checkpointing=False,
        default_root_dir=log_path,
        max_epochs=pre_epochs,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False)

    trainer.fit(model=auto_enc, datamodule=cifar10)

    lnr_memae = CIFAR10MemAESDV(chnum_in=3,
                                 seed=seed,
                                 rep_dim=200,
                                 objective=objective)
    transfer_weights(lnr_memae, auto_enc)
    lnr_memae.init_center_c(lnr_memae, cifar10.train_dataloader())
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        #  enable_checkpointing=False,
        default_root_dir=log_path,
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False)
    trainer.fit(model=lnr_memae, datamodule=cifar10)


if __name__ == '__main__':

    args = init_envir()
    cifar10_lenet(bash_log_name=args.bash_log_name,
                  normal_class=args.normal_class,
                  pre_epochs=args.pre_epochs,
                  epochs=args.epochs,
                  seed=args.seed,
                  radio=args.radio,
                  batch_size=args.batch_size,
                  enable_progress_bar=args.progress_bar,
                  log_path=args.log_path,
                  objective=args.objective,
                  devices=args.devices)
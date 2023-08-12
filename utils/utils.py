import torch
import numpy as np
import random
import os
import sys
import yaml
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

from tsne_torch import TorchTSNE as TSNE
import matplotlib.pyplot as plt


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True


def display_tsne(net, dataloader, filename='tsne.png'):
    n_samples = 0
    data_xs = []
    net = net.cuda()
    collect_labels = []
    with torch.no_grad():
        for data in dataloader:
            # get the inputs of the batch
            inputs, labels = data
            inputs = inputs.cuda()
            outputs = net(inputs)
            if isinstance(outputs, dict):
                outputs = outputs["enc_out"]
                outputs = outputs.contiguous().view(outputs.size(0), -1)
            n_samples += outputs.shape[0]
            data_xs.append(outputs)
            collect_labels.append(labels)

    data_x = torch.cat(data_xs, dim=0)
    collect_labels = torch.cat(collect_labels, dim=0).cpu().detach().numpy()
    X_emb = TSNE(n_components=2).fit_transform(data_x.cpu().numpy())
    X_emb = X_emb.cpu().detach().numpy()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=[collect_labels])
    plt.savefig(filename)


def PCA_svd(X, k, center=True):
    n = X.shape[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) *
         torch.mm(ones, ones.t())) if center else torch.zeros(n *
                                                              n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components.cpu().detach().numpy()


def transfer_weights(dst_net, src_net):
    """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

    dst_net_dict = dst_net.state_dict()
    src_net_dict = src_net.state_dict()

    # Filter out decoder network keys
    src_net_dict = {k: v for k, v in src_net_dict.items() if k in dst_net_dict}
    # Overwrite values in the existing state_dict
    dst_net_dict.update(src_net_dict)
    # Load the new state_dict
    dst_net.load_state_dict(dst_net_dict)



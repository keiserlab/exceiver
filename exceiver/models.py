#!/srv/home/wconnell/anaconda3/envs/lit-trans

"""
Author: Will Connell
Date Initialized: 2022-04-07
Email: connell@keiserlab.org

Exceiver models. Core modules based on Perceiver IO architecture: https://arxiv.org/abs/2107.14795.
"""


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                IMPORT MODULES
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


# I/O
import argparse

# Modeling
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection, ExplainedVariance
from torchmetrics import Accuracy


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                PRIMARY MODELS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


class CrossAttn(nn.Module):
    def __init__(self, query_dim, key_val_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.layernorm_kv = nn.LayerNorm(key_val_dim)
        self.layernorm_q = nn.LayerNorm(query_dim)
        self.cross_attn = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=dropout,
            kdim=key_val_dim,
            vdim=key_val_dim,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim),
        )

    def forward(self, query, key_val, key_padding_mask=None, residual=True):
        norm_k = self.layernorm_kv(key_val)
        norm_v = self.layernorm_kv(key_val)
        norm_q = self.layernorm_q(query)
        latent, weights = self.cross_attn(norm_q, norm_k, norm_v, key_padding_mask)
        # residual connection
        if residual:
            latent = latent + query
        latent = self.mlp(latent) + latent
        return latent, weights


class ProcessSelfAttn(nn.Module):
    def __init__(self, embed_dim, num_layers, nhead, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, latent):
        return self.transformer(latent)


class Exceiver(pl.LightningModule):
    def __init__(
        self,
        seq_len,
        seq_dim,
        query_len,
        query_dim,
        num_layers: int = 1,
        nhead: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,  # process attn module
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        **kwargs
    ):

        # Initialize superclass
        super().__init__()

        # Relevant hyperparameters
        self.seq_len = seq_len
        self.query_dim = query_dim
        self.seq_dim = seq_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Embeddings and attention blocks
        self.gene_emb = nn.Embedding(seq_len + 1, seq_dim, padding_idx=seq_len)
        self.query_emb = nn.Parameter(torch.randn(query_len, query_dim))
        self.encoder_cross_attn = CrossAttn(query_dim, seq_dim)
        self.process_self_attn = ProcessSelfAttn(
            query_dim, num_layers, nhead, dim_feedforward, dropout
        )
        self.decoder_cross_attn = CrossAttn(
            seq_dim, query_dim
        )  # query is now gene embedding

        # MLP for gene expression prediction
        self.mlp = nn.Sequential(
            nn.LayerNorm(seq_dim), nn.Linear(seq_dim, 1), nn.GELU(), nn.Linear(1, 1)
        )

        # Functions and metrics
        self.mse = nn.MSELoss()
        self.metrics = MetricCollection([ExplainedVariance()])
        self.train_metrics = self.metrics.clone(prefix="train_")
        self.val_metrics = self.metrics.clone(prefix="val_")
        self.test_metrics = self.metrics.clone(prefix="test_")

        # Save hyperparameters
        self.save_hyperparameters()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--seq_dim", type=int, default=4, help="Dimension of gene representations."
        )
        parser.add_argument(
            "--query_len",
            type=int,
            default=64,
            help="Size of input query, or latent representation length.",
        )
        parser.add_argument(
            "--query_dim", type=int, default=4, help="Dimension of input query."
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=1,
            help="Number of ProcessSelfAttention layers.",
        )
        parser.add_argument(
            "--nhead", type=int, default=1, help="Number of ProcessSelfAttention heads."
        )
        parser.add_argument(
            "--dim_feedforward",
            type=int,
            default=2048,
            help="Dimension of ProcessSelfAttention feedforward network.",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.0,
            help="Value of ProcessSelfAttention dropout.",
        )
        parser.add_argument(
            "--learning_rate", type=float, default=3e-4, help="Learning rate."
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-5, help="Optimizer weight decay."
        )
        parser.add_argument(
            "--warmup_steps",
            type=int,
            default=500,
            help="Number of learning rate warmup steps (linear).",
        )
        return parser

    def encoder_attn_step(self, gene_ids, gene_vals, input_query, key_padding_mask):
        gene_emb = self.gene_emb(gene_ids.long())
        key_val = gene_vals.unsqueeze(2) * gene_emb
        latent, encoder_weights = self.encoder_cross_attn(
            input_query, key_val, key_padding_mask
        )
        return latent, encoder_weights

    def forward(self, gene_ids, gene_vals, input_query, key_padding_mask, mask_ids):
        output_query = self.gene_emb(mask_ids)
        latent, encoder_weights = self.encoder_attn_step(
            gene_ids, gene_vals, input_query, key_padding_mask
        )
        latent = self.process_self_attn(latent)
        decoder_out, decoder_weights = self.decoder_cross_attn(
            output_query, latent, key_padding_mask=None
        )
        return decoder_out

    def training_step(self, batch, batch_idx):
        gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask = batch
        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)
        decoder_out = self.forward(
            gene_ids, gene_vals, input_query, key_padding_mask, mask_ids
        )
        y_hat = self.mlp(decoder_out)
        loss = self.mse(y_hat, mask_vals.unsqueeze(2))
        metrics = self.train_metrics(y_hat, mask_vals.unsqueeze(2))
        loss_dict = {"train_MSELoss": loss}
        self.log_dict(loss_dict | metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask = batch
        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)
        decoder_out = self.forward(
            gene_ids, gene_vals, input_query, key_padding_mask, mask_ids
        )
        y_hat = self.mlp(decoder_out)
        loss = self.mse(y_hat, mask_vals.unsqueeze(2))
        metrics = self.val_metrics(y_hat, mask_vals.unsqueeze(2))
        loss_dict = {"val_MSELoss": loss}
        self.log_dict(loss_dict | metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask = batch
        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)
        decoder_out = self.forward(
            gene_ids, gene_vals, input_query, key_padding_mask, mask_ids
        )
        y_hat = self.mlp(decoder_out)
        return y_hat, mask_vals

    def test_step_end(self, results):
        y_hat, mask_vals = results
        loss = self.mse(y_hat, mask_vals.unsqueeze(2))
        metrics = self.test_metrics(y_hat, mask_vals.unsqueeze(2))
        loss_dict = {"test_MSELoss": loss}
        self.log_dict(loss_dict | metrics, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1) / float(self.hparams.warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

        # update params
        optimizer.step(closure=closure)


class ExceiverClassifier(Exceiver):
    def __init__(
        self,
        seq_len,
        seq_dim,
        query_len,
        query_dim,
        classify_dim=None,
        num_layers: int = 1,
        nhead: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,  # process attn module
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        **kwargs
    ):

        # Initialize superclass
        super().__init__(
            seq_len,
            seq_dim,
            query_len,
            query_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )

        # Add an extra token to the embedding for the classification task
        self.gene_emb = nn.Embedding(seq_len + 2, seq_dim, padding_idx=seq_len)

        # MLP for classification
        self.mlp_classify = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, classify_dim),
            nn.GELU(),
            nn.Linear(classify_dim, classify_dim),
        )

        # Set up functions for classification
        self.cel = nn.CrossEntropyLoss()
        self.class_metrics = MetricCollection([Accuracy()])
        self.train_class_metrics = self.class_metrics.clone(prefix="train_")
        self.val_class_metrics = self.class_metrics.clone(prefix="val_")
        self.test_class_metrics = self.class_metrics.clone(prefix="test_")
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, gene_ids, gene_vals, input_query, key_padding_mask, mask_ids):
        classify_idx = torch.full((len(gene_ids), 1), self.seq_len + 1).to(self.device)
        output_query = self.gene_emb(torch.cat((mask_ids, classify_idx), dim=1))
        latent, encoder_weights = self.encoder_attn_step(
            gene_ids, gene_vals, input_query, key_padding_mask
        )
        latent = self.process_self_attn(latent)
        decoder_out, decoder_weights = self.decoder_cross_attn(
            output_query, latent, key_padding_mask=None
        )
        return decoder_out

    def training_step(self, batch, batch_idx):
        gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask, classify = batch
        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)
        decoder_out = self.forward(
            gene_ids, gene_vals, input_query, key_padding_mask, mask_ids
        )
        y_hat = self.mlp(decoder_out[:, : mask_ids.shape[1], :])
        classify_hat = torch.squeeze(
            self.mlp_classify(decoder_out[:, mask_ids.shape[1], :])
        )
        loss = self.mse(y_hat, mask_vals.unsqueeze(2))
        ce_loss = self.cel(classify_hat, classify)
        class_metrics = self.train_class_metrics(
            torch.argmax(classify_hat, dim=1), classify
        )
        metrics = self.train_metrics(y_hat, mask_vals.unsqueeze(2))
        loss_dict = {"train_MSELoss": loss, "train_CELoss": ce_loss}
        self.log_dict(
            loss_dict | metrics | class_metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss + ce_loss

    def validation_step(self, batch, batch_idx):
        gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask, classify = batch
        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)
        decoder_out = self.forward(
            gene_ids, gene_vals, input_query, key_padding_mask, mask_ids
        )
        y_hat = self.mlp(decoder_out[:, : mask_ids.shape[1], :])
        classify_hat = torch.squeeze(
            self.mlp_classify(decoder_out[:, mask_ids.shape[1], :])
        )
        loss = self.mse(y_hat, mask_vals.unsqueeze(2))
        ce_loss = self.cel(classify_hat, classify)
        class_metrics = self.val_class_metrics(
            torch.argmax(classify_hat, dim=1), classify
        )
        metrics = self.val_metrics(y_hat, mask_vals.unsqueeze(2))
        loss_dict = {"val_MSELoss": loss, "val_CELoss": ce_loss}
        self.log_dict(
            loss_dict | metrics | class_metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss + ce_loss

    def test_step(self, batch, batch_idx):
        gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask, classify = batch
        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)
        decoder_out = self.forward(
            gene_ids, gene_vals, input_query, key_padding_mask, mask_ids
        )
        y_hat = self.mlp(decoder_out[:, : mask_ids.shape[1], :])
        classify_hat = torch.squeeze(
            self.mlp_classify(decoder_out[:, mask_ids.shape[1], :])
        )
        return y_hat, mask_vals, classify_hat, classify

    def test_step_end(self, results):
        y_hat, mask_vals, classify_hat, classify = results
        loss = self.mse(y_hat, mask_vals.unsqueeze(2))
        ce_loss = self.cel(classify_hat, classify)
        class_metrics = self.test_class_metrics(
            torch.argmax(classify_hat, dim=1), classify
        )
        metrics = self.test_metrics(y_hat, mask_vals.unsqueeze(2))
        loss_dict = {"test_MSELoss": loss, "test_CELoss": ce_loss}
        self.log_dict(
            loss_dict | metrics | class_metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                           FINE TUNE MODELS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


class LinearBlock(nn.Module):
    """
    Credit to fastai `categorical` model.

    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = self.generate_layers(*args, **kwargs)

    def generate_layers(self, in_sz, layers, out_sz, ps, use_bn, bn_final):
        if ps is None:
            ps = [0] * len(layers)
        else:
            ps = ps * len(layers)
        sizes = self.get_sizes(in_sz, layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes) - 2)] + [None]
        layers = []
        for i, (n_in, n_out, dp, act) in enumerate(
            zip(sizes[:-1], sizes[1:], [0.0] + ps, actns)
        ):
            layers += self.bn_drop_lin(
                n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act
            )
        if bn_final:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        block = nn.Sequential(*layers)
        return block

    def get_sizes(self, in_sz, layers, out_sz):
        return [in_sz] + layers + [out_sz]

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        x = self.block(x)
        return x

"""
Author: Will Connell
Date Initialized: 2022-04-06
Email: connell@keiserlab.org

Exceiver datasets and dataloaders.
"""


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                IMPORT MODULES
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


# PyTorch
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    BatchSampler,
    SequentialSampler,
)

# Utils
import joblib
import argparse
from pathlib import Path

# scRNAseq
import scanpy as sc


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


class ExceiverDataset(Dataset):
    def __init__(
        self,
        csr,
        classes,
        scaler,
        n_mask,
        batch_size,
        max_value=10,
        inference=False,
        pin_memory=False,
    ):
        self.csr = csr
        self.csr.eliminate_zeros()
        self.classes = classes
        self.scaler = scaler
        self.n_samples = csr.shape[0]
        self.n_features = csr.shape[1]
        self.n_mask = n_mask
        self.batch_size = batch_size
        self.max_value = max_value
        self.inference = inference
        self.pin_memory = pin_memory

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: list):

        if len(idx) != self.batch_size:
            raise ValueError("Index length not equal to batch_size")

        # transform data
        gene_vals = torch.tensor(
            self.scaler.transform(np.log1p(self.csr[idx]).toarray())
        )

        # clip outliers
        gene_vals[gene_vals > self.max_value] = self.max_value
        # shift to mean of 1
        gene_vals += 1
        # embedding indices
        gene_ids = torch.arange(0, self.n_features).repeat(self.batch_size, 1)
        # input key_padding_mask to mask attention at "pad" indices
        # get nonsparse indices
        row, col = self.csr[idx].nonzero()
        key_padding_mask = torch.ones_like(gene_vals).detach()
        key_padding_mask[row, col] = 0

        # for inference
        if self.inference:
            # we want to predict each gene
            mask_ids = gene_ids.clone().detach()
            # we want to return all true values (needs subsetting w/ key_padding_mask)
            mask_vals = gene_vals.clone().detach()

        # for training
        else:

            # mask indices
            mask_col_ids = torch.tensor(
                np.concatenate(
                    [
                        np.random.choice(
                            self.csr[i].indices, self.n_mask, replace=False
                        )
                        for i in idx
                    ]
                )
            ).long()
            mask_row_ids = torch.repeat_interleave(
                torch.arange(0, self.batch_size), self.n_mask
            ).long()
            assert len(mask_col_ids) == len(mask_row_ids)
            # we basically want to predict these values from zero, the underdetermined default of this datatype
            # extract values to predict (y)
            mask_vals = gene_vals[mask_row_ids, mask_col_ids].reshape(
                self.batch_size, -1
            )
            # input key_val matrix shifted to mean value == 1
            gene_vals[mask_row_ids, mask_col_ids] = 1
            # input gene_id matrix should have "mask" index at mask indices
            # extract idx to predict (y_idx)
            mask_ids = gene_ids[mask_row_ids, mask_col_ids].reshape(self.batch_size, -1)
            # "mask" = second to last embedding index = n_features
            gene_ids[mask_row_ids, mask_col_ids] = self.n_features

        if self.classes is None:
            batch = gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask
        else:
            classes = torch.tensor(self.classes[idx])
            batch = gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask, classes

        if self.pin_memory:
            for tensor in batch:
                tensor.pin_memory()

        return batch


class ExceiverDataModule(pl.LightningDataModule):
    def __init__(self, data_path, classify, frac=0.15, batch_size=32, num_workers=6):
        super().__init__()
        self.data_path = data_path
        self.classify = classify
        self.frac = frac
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path", type=Path, help="Path to directory with preprocessed data."
        )
        parser.add_argument(
            "--classify",
            type=str,
            default=None,
            help="Name of column from `obs` table to add classification task with. (optional)",
        )
        parser.add_argument(
            "--frac",
            type=float,
            default=0.15,
            help="Fraction of median genes to mask for prediction.",
        )
        parser.add_argument(
            "--batch_size", type=int, default=32, help="Dataloader batch size."
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=6,
            help="Number of workers for DataLoader.",
        )
        return parser

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:
    def prepare_data(self):

        # Prepare expression values
        self.features = joblib.load(self.data_path.joinpath("gene-features.pkl"))
        self.n_features = len(self.features)
        self.train_scaler = joblib.load(self.data_path.joinpath("train-scaler.pkl"))
        self.val_scaler = joblib.load(self.data_path.joinpath("val-scaler.pkl"))
        self.train_adata = sc.read_h5ad(self.data_path.joinpath("train.h5ad"))
        self.val_adata = sc.read_h5ad(self.data_path.joinpath("val.h5ad"))

        # filter cells by (2 * fraction * median genes per sample)
        self.n_mask = int(
            self.train_adata.obs["n_genes_by_counts"].median() * self.frac
        )
        print(f"n masked genes: {self.n_mask}")
        sc.pp.filter_cells(self.train_adata, min_genes=self.n_mask * 2)
        sc.pp.filter_cells(self.val_adata, min_genes=self.n_mask * 2)
        print(f"train_adata shape: {self.train_adata.shape}")
        print(f"val_adata shape: {self.val_adata.shape}")

        # Prepare classification labels
        print(f"preparing class labels for {self.classify}")
        self.train_classes = None
        self.val_classes = None

        if self.classify:
            if not np.array_equal(
                np.unique(self.train_adata.obs[self.classify]),
                np.unique(self.val_adata.obs[self.classify]),
            ):
                raise ValueError(f"Class mismatch in *_adata.obs[{self.classify}]")
            self.classify_codes = list(self.train_adata.obs[self.classify].unique())
            self.classify_dict = {j: i for i, j in enumerate(self.classify_codes)}

            self.train_classes = np.array(
                [
                    self.classify_dict[code]
                    for code in self.train_adata.obs[self.classify]
                ]
            )
            self.val_classes = np.array(
                [self.classify_dict[code] for code in self.val_adata.obs[self.classify]]
            )

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        if stage == "fit":

            self.train_dataset = ExceiverDataset(
                self.train_adata.X,
                classes=self.train_classes,
                scaler=self.train_scaler,
                n_mask=self.n_mask,
                batch_size=self.batch_size,
                pin_memory=False,
            )
            self.val_dataset = ExceiverDataset(
                self.val_adata.X,
                classes=self.val_classes,
                scaler=self.val_scaler,
                n_mask=self.n_mask,
                batch_size=self.batch_size,
                pin_memory=False,
            )

            return self.train_dataset, self.val_dataset

        if stage == "test":
            raise NotImplementedError

    # return the dataloader for each split
    def train_dataloader(self, num_workers=6):
        sampler = BatchSampler(
            RandomSampler(self.train_dataset),
            batch_size=self.train_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.train_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dl

    def val_dataloader(self, num_workers=6):
        sampler = BatchSampler(
            SequentialSampler(self.val_dataset),
            batch_size=self.val_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.val_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dl

    def test_dataloader(self):
        raise NotImplementedError

"""
Author: Will Connell
Date Initialized: 2022-04-07
Email: connell@keiserlab.org
"""


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                IMPORT MODULES
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


# I/0
import sys
import joblib
from pathlib import Path
import argparse

# Data handling
import numpy as np
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import anndata as ad

from exceiver.genes import PROTCODE_GENES


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def main(args):

    # Load in Tabula Sapiens data
    tsdata = sc.read_h5ad(args.ts_path)
    print("Loaded data.")

    # Subset to protein coding genes (defined by CCLE)
    gene_features = list(set(PROTCODE_GENES).intersection(set(tsdata.var_names)))
    tsdata = tsdata[:, gene_features]
    print("Subsetted to CCLE genes.")

    # QC follows standard scanpy preprocessing tutorial
    # https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html

    # tsdata.X is scVI corrected gene matrix
    tsdata = ad.AnnData(tsdata.layers[args.count_var], obs=tsdata.obs, var=tsdata.var, uns=tsdata.uns)
    print("Selected AnnData layer.")

    # remove incorrect non-sparse idx
    tsdata.X.eliminate_zeros()
    print("Eliminated zeros.")

    # shuffle
    sc.pp.subsample(tsdata, fraction=1, random_state=0)
    print("Shuffled data.")

    # random split
    train_loc = tsdata.obs.sample(frac=0.7, replace=False).index
    val_loc = tsdata.obs.index[~tsdata.obs.index.isin(train_loc)]
    ts_train = tsdata[train_loc, :]
    ts_val = tsdata[val_loc, :]
    print("Split into training and testing.")

    # calculate metrics
    sc.pp.calculate_qc_metrics(ts_train, expr_type="counts", inplace=True)
    sc.pp.calculate_qc_metrics(ts_val, expr_type="counts", inplace=True)
    print("Calculated QC metrics.")

    # remove genes with high sparsity
    sc.pp.filter_genes(
        ts_train, min_cells=int(ts_train.n_obs * args.gene_filter), inplace=True
    )
    ts_val = ts_val[:, ts_train.var_names]
    print(f"Filtered genes using gene_filter = {args.gene_filter}.")

    # recalculate metrics
    sc.pp.calculate_qc_metrics(ts_train, expr_type="counts", inplace=True)
    sc.pp.calculate_qc_metrics(ts_val, expr_type="counts", inplace=True)
    print("Recalculated QC metrics.")

    # total count normalize
    sc.pp.normalize_total(ts_train, target_sum=1e4)
    sc.pp.normalize_total(ts_val, target_sum=1e4)
    print("Normalized counts.")

    # zscore on log1p data
    train_scaler = StandardScaler()
    val_scaler = StandardScaler()
    train_scaler.fit(np.log1p(ts_train.X).toarray())
    val_scaler.fit(np.log1p(ts_val.X).toarray())
    print("Fit scalers.")

    # write out untransformed data and scalers
    args.out_path.mkdir(parents=True, exist_ok=False)
    joblib.dump(gene_features, args.out_path.joinpath(f"gene-features.pkl"))
    joblib.dump(train_scaler, args.out_path.joinpath(f"train-scaler.pkl"))
    joblib.dump(val_scaler, args.out_path.joinpath(f"val-scaler.pkl"))
    ts_train.write(args.out_path.joinpath(f"train.h5ad"))
    ts_val.write(args.out_path.joinpath(f"val.h5ad"))
    print("Saved data.")


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                    CLI
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


if __name__ == "__main__":

    # Set up argument parser
    desc = "Script for preprocessing scRNAseq training data."
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ts_path",
        type=Path,
        help="File path to TabulaSapiens scRNAseq data."
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        help="Directory path to write processed data."
    )
    parser.add_argument(
        "--gene_filter",
        type=float,
        default=0.000,
        help="Minimum fraction of cells a gene must appear in to pass filtering.",
    )
    parser.add_argument(
        "--count_var",
        type=str,
        default="decontXcounts",
        help="Layer to select from input AnnData object."
    )

    # Run
    args = parser.parse_args()
    sys.exit(main(args))

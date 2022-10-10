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


# I/O
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Modeling
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# Custom
from exceiver.datasets import ExceiverDataModule
from exceiver.models import Exceiver, ExceiverClassifier


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                              PRIMARY FUNCTIONS                                                           
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################
    

def main(args):

    # Set seed
    seed_everything(2299)

    # Make logging directory
    args.logs.mkdir(parents=True, exist_ok=True)
    
    # Set up data module
    dm = ExceiverDataModule.from_argparse_args(args)
    dm.prepare_data()

    # Condense args
    args.seq_len = dm.n_features
    dict_args = vars(args)

    # Set up model
    if args.classify is None:
        model = Exceiver(**dict_args)
    else:
        model = ExceiverClassifier(classify_dim = dm.val_adata.obs[args.classify].nunique(), **dict_args)

    # Set up callbacks
    logger = TensorBoardLogger(save_dir=args.logs,
                               version=args.name,
                               name='lightning_logs',
                               default_hp_metric=False)
    early_stop = EarlyStopping(monitor='val_MSELoss',
                               min_delta=1e-5,
                               patience=10,
                               verbose=False,
                               mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_MSELoss',
                                          mode='min',
                                          save_last=True)
    if args.strategy == "ddp":
        pass

    # Set up Trainer
    start = datetime.now()
    trainer = Trainer.from_argparse_args(args,
                                         default_root_dir=logger.log_dir,
                                         logger=logger,
                                         callbacks=[early_stop, checkpoint_callback],
                                         profiler="simple",
                                         replace_sampler_ddp=False)

    # Train model
    trainer.fit(model, dm)
    print("Completed in {}".format(str(datetime.now() - start)))
    print(f"Model saved at {checkpoint_callback.best_model_path}")


if __name__ == '__main__':

    # Set up argument parser
    desc = "Script for self-supervised scRNAseq training."
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Prepended name of experiment.")
    parser.add_argument("--logs", type=Path, help="Path to model logs and checkpoints.")

    # Add arguments from classes
    parser = Exceiver.add_argparse_args(parser)
    parser = ExceiverDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Parse arguments and train
    args = parser.parse_args()
    sys.exit(main(args))
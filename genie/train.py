import os
import wandb
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from genie.config import Config
from genie.utils.model_io import load_model
from genie.diffusion.genie import Genie

from genie.data.ambient_data_module import AmbientDataModule


def main(args):

    # Load configuration
    config = Config(filename=args.config)

    # Create logger
    loggers = []
    if not args.test:
        loggers.append(
            TensorBoardLogger(
                save_dir=config.io['rootdir'],
                name=config.io['name']
            )
        )
        loggers.append(
            WandbLogger(
                project="ambient-protein-diffusion",
                name=config.io["name"],
                config={
                    **config.io,
                    **config.diffusion,
                    **config.model,
                    **config.training,
                    **config.optimization,
                },
                save_dir=config.io["rootdir"],
                settings=wandb.Settings(code_dir="."),
            )
        )

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config.training['checkpoint_every_n_epoch'],
        filename='{epoch}',
        save_top_k=-1
    )

    # Initial random seeds
    seed_everything(config.training['seed'], workers=True)

    # Data module
    dm = AmbientDataModule(
        **config.io,
        diffusion_config=config.diffusion,
        training_config=config.training,
        batch_size=config.training['batch_size']
    )

    # Model
    model = load_model(config.io['rootdir'], config.io['name'])
    if config.training['resume'] is not None:
        print(f'Resuming from {config.training["resume"]}')
        model = Genie.load_from_checkpoint(config.training['resume'], config=config)

    # Trainer
    trainer = Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        accelerator='gpu',
        logger=loggers,
        strategy='ddp',
        deterministic=False,
        enable_progress_bar=args.test,
        log_every_n_steps=config.training['log_every_n_step'],
        max_epochs=config.training['n_epoch'],
        gradient_clip_val=5,
        accumulate_grad_batches=config.training["accum_grad"],
        use_distributed_sampler=False,
        callbacks=[checkpoint_callback]
    )

    # Run
    trainer.fit(model, dm)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--devices', type=int, help='Number of GPU devices to use')
    parser.add_argument('-n', '--num_nodes', type=int, help='Number of nodes')
    parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
    parser.add_argument('-t', '--test', action='store_true', help='Enable test mode', default=False)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)

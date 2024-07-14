import logging
import sys

from einops import rearrange
from torcheval.metrics import MulticlassAccuracy

from datasets.dataset_name import DatasetName
from models import ModelType
from train.trainer import PilotNetTrainer, PerceiverTrainer, Trainer


def build_trainer(args):
    if args.model_type == ModelType.PILOTNET.value:
        return PilotNetTrainer(args.model_name, wandb_project=args.wandb_project)
    elif args.model_type == ModelType.PERCEIVER.value:
        is_many_to_one = False
        save_model = True
        target_name = 'steering_angle'
        metric_multi_class_accuracy = None
        prepare_dataloader_data_fn = None
        if args.dataset == DatasetName.UCF_11.value:
            is_many_to_one = True
            save_model = False
            target_name = 'n/a'
            metric_multi_class_accuracy = MulticlassAccuracy()

            def prepare_dataloader_data_fn_ucf(loader_data):
                data = loader_data
                inputs = rearrange(data['video'], 'b c t h w -> t b h w c')
                target_values = data['label']
                return inputs, target_values

            prepare_dataloader_data_fn = prepare_dataloader_data_fn_ucf

        elif args.dataset == DatasetName.RALLY_ESTONIA.value:

            def prepare_dataloader_data_fn_rally_estonia(loader_data):
                data, target_values, _ = loader_data
                inputs = rearrange(data['image'], 'b t c h w -> t b h w c')  # (T, B, H, W, C)
                target_values = rearrange(target_values, 'b t -> t b')  # (T, B)
                return inputs, target_values

            prepare_dataloader_data_fn = prepare_dataloader_data_fn_rally_estonia

        elif args.dataset == DatasetName.COCO_17.value:
            return Trainer(
                model_name=args.model_name,
                wandb_project=args.wandb_project,
                target_name=target_name,
                save_model=save_model,
                metric_multi_class_accuracy=metric_multi_class_accuracy,
            )

        return PerceiverTrainer(
            model_name=args.model_name,
            wandb_project=args.wandb_project,
            target_name=target_name,
            save_model=save_model,
            metric_multi_class_accuracy=metric_multi_class_accuracy,
            prepare_dataloader_data_fn=prepare_dataloader_data_fn
        )
    else:
        logging.error("Unknown model type: %s", args.model_type)
        sys.exit()

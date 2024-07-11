from datasets import get_coco_api_from_dataset
from datasets.dataset_name import DatasetName
from metrics.coco_eval import CocoEvaluator
from metrics.rally_estonia_eval import RallyEstoniaEval


def get_build_evaluators_fn(args, val_dataset):
    def build_evaluators():
        evaluators = []
        if args.dataset == DatasetName.COCO_17.value:
            evaluators.append(CocoEvaluator(
                coco_gt=get_coco_api_from_dataset(val_dataset),
                iou_types=['bbox'])
            )
        elif args.dataset == DatasetName.RALLY_ESTONIA.value:
            evaluators.append(
                RallyEstoniaEval("steering_angle", val_dataset)
            )
        elif args.dataset == DatasetName.UCF_11.value:
            evaluators.append(
                RallyEstoniaEval("steering_angle", val_dataset)
            )
        return evaluators
    return build_evaluators
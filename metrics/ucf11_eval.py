from torcheval.metrics import MulticlassAccuracy


class Ucf11Eval:

    def __init__(self):
        self.metric_multi_class_accuracy = MulticlassAccuracy()

    def update(self, postprocessors, loader_data, predictions):
        self.metric_multi_class_accuracy.update(
            predictions.squeeze(), loader_data['label']
        )

    def fill_stats(self, stats):
        stats['ucf11_multi_class_accuracy'] = self.metric_multi_class_accuracy.compute().item()
        self.metric_multi_class_accuracy.reset()
        return stats

from metrics.metrics import calculate_open_loop_metrics


class RallyEstoniaEval:

    def __init__(self, target_name, valid_dataset):
        self.all_predictions = []
        self.target_name = target_name
        self.valid_dataset = valid_dataset
        self.fps = 30

    def update(self, postprocessors, loader_data, predictions):
        self.all_predictions.extend(predictions.cpu().squeeze().numpy())

    def fill_stats(self, stats):
        metrics = self.calculate_metrics(self.fps, self.all_predictions, self.valid_dataset)
        stats['rally_estonia_eval'] = metrics
        self.all_predictions = []
        return stats

    def calculate_metrics(self, fps, predictions, valid_dataset):
        if self.target_name == "steering_angle":
            frames_df = valid_dataset.frames
            true_steering_angles = frames_df.steering_angle.to_numpy()
            metrics = calculate_open_loop_metrics(predictions, true_steering_angles, fps=fps)
            left_turns = frames_df["turn_signal"] == 0
            if left_turns.any():
                left_metrics = calculate_open_loop_metrics(predictions[left_turns], true_steering_angles[left_turns],
                                                           fps=fps)
                metrics["left_mae"] = left_metrics["mae"]
            else:
                metrics["left_mae"] = 0

            straight = frames_df["turn_signal"] == 1
            if straight.any():
                straight_metrics = calculate_open_loop_metrics(predictions[straight], true_steering_angles[straight],
                                                               fps=fps)
                metrics["straight_mae"] = straight_metrics["mae"]
            else:
                metrics["straight_mae"] = 0

            right_turns = frames_df["turn_signal"] == 2
            if right_turns.any():
                right_metrics = calculate_open_loop_metrics(predictions[right_turns], true_steering_angles[right_turns],
                                                            fps=fps)
                metrics["right_mae"] = right_metrics["mae"]
            else:
                metrics["right_mae"] = 0
        else:
            metrics = {}

        return metrics
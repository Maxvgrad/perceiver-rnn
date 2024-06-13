import numpy as np


def calculate_open_loop_metrics(predicted_steering, true_steering, fps):
    predicted_degrees = predicted_steering / np.pi * 180
    true_degrees = true_steering / np.pi * 180
    errors = np.abs(true_degrees - predicted_degrees)
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    max = errors.max()

    whiteness = calculate_whiteness(predicted_degrees, fps)
    expert_whiteness = calculate_whiteness(true_degrees, fps)

    return {
        'mae': mae,
        'rmse': rmse,
        'max': max,
        'whiteness': whiteness,
        'expert_whiteness': expert_whiteness
    }


def calculate_whiteness(steering_angles, fps=30):
    current_angles = steering_angles[:-1]
    next_angles = steering_angles[1:]
    delta_angles = next_angles - current_angles
    whiteness = np.sqrt(((delta_angles * fps) ** 2).mean())
    return whiteness

"""Simple schedules used in controlled graph training experiments."""
import math


def linear_to_zero(step, initial_value, stop_update):
    if stop_update < 2 or step < 1:
        raise ValueError('Schedule requires a positive step and stop_update >= 2')
    return initial_value * max(0., 1. - (step - 1) / (stop_update - 1))


def annealed_encoder_lr(step, initial_lr, stop_update):
    return linear_to_zero(step, initial_lr, stop_update)


def cosine_to_zero(step, initial_lr, total_updates):
    if step < 1 or total_updates < 2:
        raise ValueError('Cosine schedule requires a positive step and at least two updates')
    progress = min((step - 1) / (total_updates - 1), 1.)
    return initial_lr * .5 * (1. + math.cos(math.pi * progress))

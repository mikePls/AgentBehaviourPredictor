import numpy as np


def get_distance_probabilities(current_point: np.array, objects: list):
    distances = {}
    scores = {}
    total_score = 0

    # Calculate distances
    for o in objects:
        distance = np.linalg.norm(o.position - current_point)
        distances[o.object_id] = distance

    # Convert distances to inverted scores (lower distance -> higher score)
    for object_id, distance in distances.items():
        # Invert the distance. Adding a small constant to avoid division by zero.
        score = 1 / (distance + 1e-6)
        scores[object_id] = score
        total_score += score

    # Calculate probabilities based on scores
    probabilities = {object_id: score / total_score for object_id, score in scores.items()}
    return probabilities


def get_uniform_pick_chance(objects, visible_objects: list) -> dict:

    uniform_prob = 1 / (len(visible_objects) +1e-16)
    probs_dict = {}
    for obj in objects:
        probs_dict[obj.object_id] = uniform_prob if obj.object_id in visible_objects else 0
    return probs_dict


def get_grip_probs(objects: list, grip_val: int, baseline_diff=0.01):
    differences = {}
    total = 0
    for obj in objects:
        dif = 1 / (np.abs(grip_val - obj.size) + baseline_diff)  # Added baseline_diff to the denominator
        differences[obj.object_id] = dif
        total += dif

    scores = {k: v / total for k, v in differences.items()}
    return scores



def get_total_probabilities(objects, current_point, visible_objects: list, grip_val: int):
    dist_probs = get_distance_probabilities(current_point, objects)

    grip_probs = get_grip_probs(objects, grip_val)

    visible_uniform = get_uniform_pick_chance(objects, visible_objects)

    total_probs = {o.object_id: ((dist_probs[o.object_id] + grip_probs[o.object_id] +
                                  visible_uniform[o.object_id])) / 3 for o in objects}

    probs_dict = {'Probabilities according to distance': dist_probs,
                  'Probabilities according to grip setting': grip_probs,
                  'Probabilities according to number of visibles': visible_uniform,
                  'Total probabilities': total_probs
                  }

    return probs_dict

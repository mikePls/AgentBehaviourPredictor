import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import probs_calculator as prc
from main_sim import Object
import os


for i  in range(1, 101):
    with open(f'saved_data/state_data{i}.pkl', 'rb') as f:
        state_data = pickle.load(f)
    with open(f'saved_data/objects{i}.pkl', 'rb') as f:
        objects = pickle.load(f)

    probs_data = {key: prc.get_total_probabilities(objects,step['position'],
                                                   step['visible_objects'],
                                              step['grip_val']) for key,step in state_data.items()}
    probs_data['target'] = state_data[0]['target']

    with open(f'saved_data/probs_data{i}.pkl', 'wb') as f:
        objects = pickle.dump(probs_data, f)
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate

import probs_calculator as prc
from main_sim import Object

with open('saved_data/state_data5.pkl', 'rb') as f:
    state_data = pickle.load(f)

with open('saved_data/objects5.pkl', 'rb') as f:
    objects = pickle.load(f)

object_names = [o.object_id for o in objects]

probs_data = {key: prc.get_total_probabilities(objects,step['position'],
                                               step['visible_objects'],
                                          step['grip_val']) for key,step in state_data.items()}

print(probs_data)
image_path = 'robot2d.png'  # Replace this with the path to your PNG image
image = plt.imread(image_path)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))


def init():
    """Initial plot setup."""
    ax1.clear()
    ax2.clear()

    ax1.set_title("Dynamic Probabilities Update")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_xlabel("Objects")
    ax1.set_xticks(range(len(object_names)))
    ax1.set_xticklabels(object_names, rotation=45, ha="right")

    ax2.set_title("2D State Representation")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")


def update(frame):
    """Update function for the animation, stacking probabilities for each object."""
    ax1.clear()  # Clear previous content
    ax2.clear()

    ax1.set_xticks(range(len(object_names)))
    ax1.set_xticklabels(object_names, rotation=45, ha="right")

    # Categories based on sample data
    categories = ['Probabilities according to grip setting',
                  'Probabilities according to distance',
                  'Probabilities according to number of visibles']

    global state_data
    true_target = state_data[frame]['target']
    gaze_angle = state_data[frame]['gaze_angle']
    visible_objects = state_data[frame]['visible_objects']

    # Initialise bottoms for all objects
    bottoms = np.zeros(len(object_names))
    has_labels = False  # Track if any bars have labels
    total_probs = probs_data[frame]['Total probabilities']

    for category in categories:
        # Initialize values for all objects to zero
        values = np.zeros(len(object_names))

        # Update values for objects that have data in this category
        for obj_index, obj_name in enumerate(object_names):
            if obj_name in probs_data[frame][category]:
                values[obj_index] = probs_data[frame][category][obj_name] /3
                has_labels = True  # We have at least one label

        ax1.bar(object_names, values, bottom=bottoms, alpha=0.5, label=category)
        bottoms += values  # Update bottoms for the next category

    for idx, obj_name in enumerate(object_names):
        if obj_name in total_probs:
            ax1.text(idx, bottoms[idx], f'{total_probs[obj_name]:.2f}', ha='center', va='bottom')

    ax1.set_ylim(0, 1)
    ax1.set_title(f"Dynamic Probabilities Update - Time Step: {frame}\n True Target: {true_target}")
    if has_labels:
        ax1.legend(loc='upper right')  # Only call legend if we have labels

    current_position = state_data[frame]['position']

    ax2.plot(current_position[0], current_position[1], 'o', markersize=10, color='red')
    for obj in objects:
        colour = 'green' if obj.object_id in visible_objects else 'blue'
        ax2.plot(obj.position[0], obj.position[1], 'o', color=colour)
        ax2.text(obj.position[0] + 2, obj.position[1], f'{total_probs[obj.object_id]:.2f}', ha='center', va='bottom')

    # Calculate and plot the robot's gaze cone
    target_position = [obj for obj in objects if obj.object_id == true_target][0].position
    direction_vector = np.array(target_position) - np.array(current_position)
    gaze_angle_rad = np.radians(gaze_angle / 2)  # Half the gaze angle in radians

    # Unit direction vector to the target
    unit_direction = direction_vector / np.linalg.norm(direction_vector)

    # Rotate the direction vector by +gaze_angle/2 and -gaze_angle/2 to get the cone edges
    rotation_matrix_1 = np.array([[np.cos(gaze_angle_rad), -np.sin(gaze_angle_rad)],
                                  [np.sin(gaze_angle_rad), np.cos(gaze_angle_rad)]])
    rotation_matrix_2 = np.array([[np.cos(-gaze_angle_rad), -np.sin(-gaze_angle_rad)],
                                  [np.sin(-gaze_angle_rad), np.cos(-gaze_angle_rad)]])

    edge_1 = np.dot(rotation_matrix_1, unit_direction[:2])
    edge_2 = np.dot(rotation_matrix_2, unit_direction[:2])

    # Plotting the gaze cone edges as dotted lines
    length = 50  # Length of the gaze lines
    ax2.plot([current_position[0], current_position[0] + edge_1[0] * length],
             [current_position[1], current_position[1] + edge_1[1] * length], 'r--')
    ax2.plot([current_position[0], current_position[0] + edge_2[0] * length],
             [current_position[1], current_position[1] + edge_2[1] * length], 'r--')

    ax2.set_title("2D Representation - Time Step: {}".format(frame))
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")

    global image
    direction_vector = np.abs(np.array(target_position) - np.array(current_position))

    # Calculate the angle between the direction vector and the horizontal axis
    angle_to_horizontal_rad = np.arctan2(direction_vector[1], direction_vector[0])

    # Adjust the angle so the top of the image faces the target position
    angle_to_horizontal_rad -= np.pi/2

    # Rotate the image by the calculated angle
    # Convert the angle from radians to degrees
    rotated_image = rotate(image, np.degrees(angle_to_horizontal_rad), reshape=True)

    # Ensure the image data is within the valid range after rotation
    rotated_image = np.clip(rotated_image, 0, 1)

    # OffsetImage with the rotated image
    imagebox = OffsetImage(rotated_image, zoom=0.1)

    # AnnotationBbox with the OffsetImage
    ab = AnnotationBbox(imagebox, (current_position[0], current_position[1]), frameon=False)


    # Add the annotation box to the plot
    ax2.add_artist(ab)


# Run the animation
ani = FuncAnimation(fig, update, frames=sorted(probs_data.keys()), init_func=init, interval=1000, repeat=0)

plt.show()

import numpy as np
import random
import threading
import pandas as pd
import threading
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import pickle


class Object:
    """ Simple Class to represent an Object/item """
    def __init__(self, position, object_id, size=10):
        self.position = np.array(position)
        self.object_id = object_id
        self.size = size


class Robot:
    def __init__(self, position=np.zeros(3), velocity=np.zeros(3), gaze_angle=45, objects=[]):
        self.position = position
        self.velocity = velocity
        self.trajectory_coeffs = {'x': None, 'y': None, 'z': None}
        self.gaze_angle = gaze_angle  # degrees
        self.objects = objects
        self.target = objects[np.random.randint(0, len(objects))] if objects else None
        self.grip_setting = 10
        self.state_history = {}
        self.ml_data = {}
        self.subscribers = []  # List of functions to notify on state update
        print(f'Target set on "{self.target.object_id}"')
        #print(self.target.position)

    def plan_trajectory(self, T):
        """Plans a trajectory using cubic splines, given target position and velocity."""
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            p0 = self.position[i]
            pf = self.target.position[i] - 1 if i !=2 else self.target.position[i]
            v0 = self.velocity[i]
            vf = 0
            a0 = p0
            a1 = v0
            a2 = (3 * (pf - p0) / T ** 2) - (2 * v0 / T) - (vf / T)
            a3 = -(2 * (pf - p0) / T ** 3) + (v0 + vf) / T ** 2
            self.trajectory_coeffs[axis] = (a0, a1, a2, a3)

    def update_position(self, t):
        """Updates the robot's position along the planned trajectory at time t."""
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            a0, a1, a2, a3 = self.trajectory_coeffs[axis]
            self.position[i] = a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0

    def get_objects_in_gaze(self):
        """Filters objects based on whether they are within the gaze angle centered on the target."""
        visible_objects = []
        # Calculate the gaze direction vector as the direction from the robot to the target
        gaze_direction = self.target.position - self.position
        for obj in self.objects:
            obj_direction = obj.position - self.position
            angle = self.calculate_angle(gaze_direction, obj_direction)
            if angle <= self.gaze_angle / 2:  # Gaze angle to each side from the direction to the target
                visible_objects.append(obj)
        return visible_objects

    def calculate_angle(self, v1, v2):
        """Calculates the angle between two vectors in degrees."""
        unit_v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) else v1
        unit_v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) else v2
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)  # Clipping for numerical stability
        angle = np.arccos(dot_product) * (180 / np.pi)  # Convert to degrees
        return angle

    def broadcast_state(self, objects):
        """Broadcasts the robot's position and the list of objects within its visual field."""
        visible_objects = self.get_objects_in_gaze()
        visible_object_ids = [obj.object_id for obj in visible_objects]
        state = {
            'position': self.position.tolist(),  # Convert numpy array to list for serialization
            'visible_objects': visible_object_ids,
            'objects': self.objects,
            'grip_val': self.grip_setting,
            'target':self.target.object_id,
            'gaze_angle': self.gaze_angle
        }

        return state

    def dataset_for_ml(self):
        """Stores datasets for machine learning analysis, structured by timestep with detailed object information, including the target."""
        # Object information
        objects_info = {
            obj.object_id: {
                'dist_from_agent': np.linalg.norm(self.position - obj.position),  # Euclidean distance
                'grip_difference': abs(self.grip_setting - obj.size),
                # Absolute difference in grip setting and object size
                'is_visible': 1 if obj in self.get_objects_in_gaze() else 0  # 1 if visible, 0 otherwise
            } for obj in self.objects
        }
        # Including target information within the same timestep
        self.ml_data[self.current_time] = {
            'objects': objects_info,
            'target': self.target.object_id  # Adding target object ID for ground truth
        }

    def execute(self, interval=1, T=5):
        """Executes the process in a loop with specified time intervals."""
        self.T = T
        self.interval = interval
        self.current_time = 0
        self.rand_grip_time = random.randint(5, T - 4)
        self.grip_changed = False
        self.plan_trajectory(T)

        while self.current_time <= self.T and not self.current_time > 200:
            self.execute_step()
            #self.dataset_for_ml() # Uncomment to gather alternative datasets
            time.sleep(0)  # Wait for 'interval' seconds before the next iteration
            self.current_time += self.interval

        print("Execution completed")

    def execute_step(self):
        """Executes a single step of the simulation."""
        self.update_position(self.current_time)
        state = self.broadcast_state(self.objects)
        self.state_history[self.current_time] = state
        print(state)
        if not self.grip_changed and self.current_time >= self.rand_grip_time:
            print('Grip changed')
            self.adjust_grip()
            self.grip_changed = True

    def adjust_grip(self):
        """Sets grip opening to a random value between object size and max grip setting."""
        grip_val = random.randint(self.target.size-1, self.target.size+1)
        self.grip_setting = grip_val


if __name__ == '__main__':
    for exp in range(1,101):
        # Create a list of objects for the robot to interact with
        objects = [Object(position=(random.randint(10, 100), random.randint(10, 100), 0),
                          size=random.randint(30, 50),
                          object_id=f'Object{i+1}') for i in range(15)]

        # Instantiate the robot
        robot = Robot(position=np.array([0, 0, 0]), velocity=np.array([1, 1, 0]), objects=objects, gaze_angle=90)

        # Execute the robot's command sequence
        robot.execute(interval=2, T=30)


        with open(f'saved_data/state_data{exp}.pkl', 'wb') as f:
            pickle.dump(robot.state_history, f)
        with open(f'saved_data/objects{exp}.pkl', 'wb') as f:
            pickle.dump(robot.objects, f)
        # with open(f'saved_data/ml_data{exp}.pkl', 'wb') as f:
        #     pickle.dump(robot.ml_data, f)
# This example computes smallest circles for point clouds of different size and
# measures execution time. Corresponding plot shows that execution times are
# linear in the number of points.

# To make this example work you have to install following packages:
# - matplotlib

import matplotlib.pyplot as plt
import numpy as np
import secots
import time

# area where to randomly place points
LON_MIN, LON_MAX = 0, 90
LAT_MIN, LAT_MAX = -30, 30

# maximum number of points
N_POINTS = 10000

# number of points to add per step
POINTS_PER_STEP = 1000

# how often to repeat calculations for a point cloud
PERMUTATIONS = 10

# save plotted data to file (empty string for no save)
SAVE_FILE = 'time.npz'

# generate points
rng = np.random.default_rng(0)
points = rng.uniform((LON_MIN, LAT_MIN), (LON_MAX, LAT_MAX), (N_POINTS, 2))

# measure execution times
steps = N_POINTS // POINTS_PER_STEP
points_for_step = np.zeros(steps, dtype=int)
time_for_step = np.zeros((steps, PERMUTATIONS), dtype=float)
n = 0
for step in range(steps):
    print(f'step {step}/{steps}\r', end='')

    n += POINTS_PER_STEP
    points_for_step[step] = n
    step_points = points[:n, :]

    for perm in range(PERMUTATIONS):
        step_points = rng.permutation(step_points, axis=0)

        start = time.process_time_ns()
        secots.smallest_circle(step_points, hemi_test=False)
        time_for_step[step, perm] = (time.process_time_ns() - start) / 1e6

# save data to file
if SAVE_FILE != '':
    np.savez(
        SAVE_FILE,
        points_for_step=points_for_step,
        time_for_step=time_for_step
    )

# plot
fig, ax = plt.subplots()
ax.plot(points_for_step, time_for_step.min(axis=1), 'o-b', label='min')
ax.plot(points_for_step, time_for_step.mean(axis=1), 'o-m', label='mean')
ax.plot(points_for_step, time_for_step.max(axis=1), 'o-r', label='max')
ax.set_xlabel('number of points')
ax.set_ylabel('time in ms')
ax.legend()
plt.show()

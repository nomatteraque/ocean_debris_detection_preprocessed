from ultralytics import YOLO
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import scipy.stats as stats

# Store each AP result per picture per pre-processing method in different arrays
# and shove them in a two tailes t.test to see if they are significantly different

original = [0.746, 0.838, 0.609, 0.893, 0.626, 0.712, 0.873, 0.937, 0.908,
            0.937, 0.937, 0.907, 0.885, 0.969, 0.995, 0.791, 0.921, 0.79, 0.968,
            0.826, 0.88, 0.974]
ucm_array = [0.863, 0.824, 0.833, 0.865, 0.651, 0.864, 0.919, 0.957, 0.913,
               0.975, 0.899, 0.906, 0.995, 0.939, 0.946, 0.665, 0.815, 0.811,
               0.827, 0.801, 0.9, 0.889]
clahe_array = [0.74, 0.688, 0.735, 0.826, 0.697, 0.674, 0.788, 0.927, 0.928,
               0.868, 0.897, 0.894, 0.895, 0.965, 0.829, 0.588, 0.824, 0.523,
               0.658, 0.785, 0.859, 0.842]
icbm_array = [0.664, 0.724, 0.515, 0.844, 0.477, 0.835, 0.835, 0.93, 0.886,
              0.819, 0.933, 0.883, 0.995, 0.89, 0.881, 0.564, 0.938, 0.646,
              0.609, 0.838, 0.857, 0.942]


original_living = [0.746, 0.838, 0.609, 0.893, 0.626, 0.712]
ucm_living = [0.863, 0.824, 0.833, 0.865, 0.651, 0.864]
clahe_living = [0.74, 0.688, 0.735, 0.826, 0.697, 0.674]
icbm_living = [0.664, 0.724, 0.515, 0.844, 0.477, 0.835]

original_living2 = [0.746, 0.838, 0.609, 0.893, 0.626, 0.712, 0.873]
ucm_living2 = [0.863, 0.824, 0.833, 0.865, 0.651, 0.864, 0.919]
clahe_living2 = [0.74, 0.688, 0.735, 0.826, 0.697, 0.674, 0.788]
icbm_living2 = [0.664, 0.724, 0.515, 0.844, 0.477, 0.835, 0.835]

original_dead = [0.908,
                 0.937, 0.937, 0.907, 0.885, 0.969, 0.995, 0.791, 0.921, 0.79, 0.968,
                 0.826, 0.88, 0.974]
ucm_dead = [0.913,
            0.975, 0.899, 0.906, 0.995, 0.939, 0.946, 0.665, 0.815, 0.811,
            0.827, 0.801, 0.9, 0.889]
clahe_dead = []
icbm_dead = []

print(stats.ttest_ind(original, ucm_array))

print(stats.ttest_ind(clahe_array, ucm_array))

print(stats.ttest_ind(icbm_array, ucm_array))

print(stats.ttest_ind(original, clahe_array))

print('\n')

print(stats.ttest_ind(original_living, ucm_living))

print(stats.ttest_ind(clahe_living, ucm_living))

print(stats.ttest_ind(icbm_living, ucm_living))

print(stats.ttest_ind(original_living, clahe_living))

print('\n')

print(stats.ttest_ind(original_living2, ucm_living2))

print(stats.ttest_ind(clahe_living2, ucm_living2))

print(stats.ttest_ind(icbm_living2, ucm_living2))

print(stats.ttest_ind(original_living2, clahe_living2))

print(sum(ucm_living2)/len(ucm_living2))

print(sum(original_living2)/len(original_living2))

print(sum(ucm_dead)/len(ucm_dead))

print(sum(original_dead)/len(original_dead))

print(stats.ttest_ind(original_dead, ucm_dead))
"""
fun_array = []

all_arrays = [original, fun_array, ucm_array, icbm_array, clahe_array]

j = 4
for i in range(len(all_arrays)):
    for t in range(i+1, i+j):
        print(stats.ttest_ind(all_arrays[i], all_arrays[t]))

    j -= 1
    if j == 0:
        break
"""
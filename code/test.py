import matplotlib.pyplot as plt
import numpy as np
import segmentation_tests as st
import segmentation as seg

# X, Y, C, D = st.small_samples_distance_test()
# st.minimum_distance_test(X, Y, C, D)

st.nn_classifier_test_brains(testDice=True)

plt.show()
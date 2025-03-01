import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import array

# Your coordinates
coordinates = [{'left_ankle': array([[296., 285.],
                                     [817., 817.]]),
                'left_elbow': array([[347., 502.],
                                     [866., 877.]]),
                'left_eye': array([[296., 287.],
                                   [817., 832.]]),
                'left_hip': array([[650., 763.],
                                   [878., 874.]]),
                'left_knee': array([[347., 296.],
                                    [866., 817.]]),
                'left_shoulder': array([[442., 468.],
                                        [845., 771.]]),
                'left_wrist': array([[502., 648.],
                                     [877., 873.]]),
                'neck': array([[347., 348.],
                               [866., 900.]]),
                'nose': array([[347., 344.],
                               [866., 831.]]),
                'right_ankle': array([[508., 650.],
                                      [920., 878.]]),
                'right_ear': array([[287., 297.],
                                    [832., 870.]]),
                'right_elbow': array([[404., 438.],
                                      [762., 691.]]),
                'right_eye': array([[-1., -1.],
                                    [-1., -1.]]),
                'right_hip': array([[648., 758.],
                                    [873., 876.]]),
                'right_knee': array([[347., 508.],
                                     [866., 920.]]),
                'right_shoulder': array([[344., 404.],
                                         [831., 762.]]),
                'right_wrist': array([[348., 442.],
                                      [900., 845.]])},
               {'left_ankle': array([[248., 240.],
                                     [539., 534.]]),
                'left_elbow': array([[255., 370.],
                                     [578., 577.]]),
                'left_eye': array([[248., 239.],
                                   [539., 545.]]),
                'left_hip': array([[482., 575.],
                                   [576., 583.]]),
                'left_knee': array([[255., 248.],
                                    [578., 539.]]),
                'left_shoulder': array([[318., 348.],
                                        [581., 516.]]),
                'left_wrist': array([[370., 479.],
                                     [577., 569.]]),
                'neck': array([[255., 258.],
                               [578., 614.]]),
                'nose': array([[255., 254.],
                               [578., 538.]]),
                'right_ankle': array([[375., 482.],
                                      [625., 576.]]),
                'right_ear': array([[239., 225.],
                                    [545., 572.]]),
                'right_elbow': array([[286., 322.],
                                      [485., 431.]]),
                'right_eye': array([[240., 228.],
                                    [534., 531.]]),
                'right_hip': array([[479., 570.],
                                    [569., 580.]]),
                'right_knee': array([[255., 375.],
                                     [578., 625.]]),
                'right_shoulder': array([[254., 286.],
                                         [538., 485.]]),
                'right_wrist': array([[258., 318.],
                                      [614., 581.]])},
               {'left_ankle': array([[280.,  270.],
                                     [1050., 1047.]]),
                'left_elbow': array([[295.,  405.],
                                     [1095., 1075.]]),
                'left_eye': array([[280.,  269.],
                                   [1050., 1058.]]),
                'left_hip': array([[542.,  642.],
                                   [1078., 1062.]]),
                'left_knee': array([[295.,  280.],
                                    [1095., 1050.]]),
                'left_shoulder': array([[373.,  372.],
                                        [1100., 1052.]]),
                'left_wrist': array([[405.,  375.],
                                     [1075.,  991.]]),
                'neck': array([[295.,  302.],
                               [1095., 1131.]]),
                'nose': array([[295.,  286.],
                               [1095., 1055.]]),
                'right_ankle': array([[428.,  542.],
                                      [1116., 1078.]]),
                'right_ear': array([[269.,  260.],
                                    [1058., 1092.]]),
                'right_elbow': array([[331.,  351.],
                                      [1000.,  928.]]),
                'right_eye': array([[-1., -1.],
                                    [-1., -1.]]),
                'right_hip': array([[375., 546.],
                                    [991., 983.]]),
                'right_knee': array([[295.,  428.],
                                     [1095., 1116.]]),
                'right_shoulder': array([[286.,  331.],
                                         [1055., 1000.]]),
                'right_wrist': array([[302.,  373.],
                                      [1131., 1100.]])},
               {'left_ankle': array([[319., 308.],
                                     [280., 277.]]),
                'left_elbow': array([[322., 413.],
                                     [347., 371.]]),
                'left_eye': array([[319., 307.],
                                   [280., 288.]]),
                'left_hip': array([[597., 721.],
                                   [336., 346.]]),
                'left_knee': array([[322., 319.],
                                    [347., 280.]]),
                'left_shoulder': array([[390., 420.],
                                        [340., 258.]]),
                'left_wrist': array([[413., 383.],
                                     [371., 247.]]),
                'neck': array([[322., 325.],
                               [347., 377.]]),
                'nose': array([[322., 316.],
                               [347., 316.]]),
                'right_ankle': array([[429., 597.],
                                      [419., 336.]]),
                'right_ear': array([[307., 296.],
                                    [288., 328.]]),
                'right_elbow': array([[358., 392.],
                                      [219., 125.]]),
                'right_eye': array([[-1., -1.],
                                    [-1., -1.]]),
                'right_hip': array([[383., 498.],
                                    [247., 167.]]),
                'right_knee': array([[322., 429.],
                                     [347., 419.]]),
                'right_shoulder': array([[316., 358.],
                                         [316., 219.]]),
                'right_wrist': array([[325., 390.],
                                      [377., 340.]])}]

# Load the image
img = mpimg.imread('test.jpg')

# Create a figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(img)

# For each frame
for frame in coordinates:
    # Plot each line
    for key in frame:
        ax.plot(frame[key][1], frame[key][0])

# Show the plot
plt.show()

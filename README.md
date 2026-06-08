Final IOU: 0.7563543319702148

Second testing: Added mixed precision to speed up training time.
implement a clamp on the weights as I noticed that the borders were too thick by the model. 
Implemented data augmentation of color jitter, and randomhorizontal flip.
Shows the Jaccard index of each class to better diagnose the errors.

Final IOU:  ~0.75-0.76

Third testing: 
Added further data augmentation with randomresizecrop and randomaffine for rotations, cropping, and translations.
Added a custom FocalIOU to better match the scenario as the pixel boundary in class 3 of the mask is very thin.
Switched from cosineannealing lr scheduler to oneshot following the idea of super-convergence (https://arxiv.org/abs/1708.07120)
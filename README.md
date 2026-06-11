# Oxford-IIIT Pet Segmentation

A U-Net from scratch to do semantic segmentation on the Oxford-IIIT Pet dataset. The masks have 3 classes: the pet, the background, and the thin boundary outline around the pet.

## Results

| Run | Mean IoU |
|-----|----------|
| 1 | 0.756 |
| 2 | ~0.75-0.76 |
| 3 | 0.777 |
| 4 | 0.791 |

## Changes per run

**Run 1** — first working version.

**Run 2**
- Added mixed precision to make training faster
- Clamped the class weights because the model was drawing the borders way too thick
- Added color jitter and random horizontal flip for augmentation
- Started printing the Jaccard index per class to debug

**Run 3**
- More augmentation. Random resized crop and random affine for rotation, cropping, and translation.
- Implemented focal loss + disce loss functions as it suits the topic
- Swapped the cosine annealing scheduler for one cycle following super convergence idea (https://arxiv.org/abs/1708.07120).

**Run 4**
- Increased the class weight of the borders as the borders were too thick in the images
- Added weight decay for overfitting
- Added test time augmentation

## Notes

The boundary class is the hard part. The pet and background classes score in the high 0.8s / 0.9s, but the thin boundary ring is stuck around 0.5-0.6 and that's what holds the mean down. 

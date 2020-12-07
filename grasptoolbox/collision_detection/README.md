# Collision Detection

Provide two types of collision detection:
1. model-free scenes with raw point clouds
2. model-based scenes with object labels

Currently only model-free collision detector is implemented.

## Example Usage
```python
mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
collision_mask, iou_list = mfcdetector.detect(grasp_group,
        approach_dist=0.03, collision_thresh=0.05, return_ious=True)
collision_mask, empty_mask = mfcdetector.detect(
        grasp_group, approach_dist=0.03, collision_thresh=0.05,
        return_empty_grasp=True, empty_thresh=0.01)
collision_mask, empty_mask, iou_list = mfcdetector.detect(
        grasp_group, approach_dist=0.03, collision_thresh=0.05,
        return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
```
See ``demo.py`` for more details.
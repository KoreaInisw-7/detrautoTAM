pose_estimation
tracker.py
tracking.py
inference_utils.py에서 generate_colors_dict
interactive_utils.py에서 index_numpy_to_one_hot_torch

iou_thresh = 0.05
yolo_every = 30
-> 30으로 나눠지는 프레임마다 새로운 사람 있는지 탐지

frames_to_propagate = 600
-> 몇 프레임까지 진행할 건지, 이 숫자가 짧으면 영상이 짤린다.

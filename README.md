pose_estimation <br>
tracker.py <br>
tracking.py <br>
inference_utils.py에서 generate_colors_dict <br>
interactive_utils.py에서 index_numpy_to_one_hot_torch <br>

iou_thresh = 0.05 <br>
yolo_every = 30 <br>
-> 30으로 나눠지는 프레임마다 새로운 사람 있는지 탐지 <br>

frames_to_propagate = 600 <br>
-> 몇 프레임까지 진행할 건지, 이 숫자가 짧으면 영상이 짤린다.


$env:KMP_DUPLICATE_LIB_OK="TRUE"
python tracking.py --video_path=C:\\deeplearning\\AutoTrackAnything\\dataset\\two.mp4 --width=960 --height=540 --frames_to_propagate=600 --output_video_path=C:\\deeplearning\\AutoTrackAnything\\result\\result.mp4 --device=0 --person_conf=0.6 --iou_thresh=0.15 --yolo_every=2 --output_path=C:\\deeplearning\\AutoTrackAnything\\result\\OUTPUT_CSV_PATH.csv

# -*- coding: utf-8 -*-
import csv
import os
import sys
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import (DEVICE, INFERENCE_SIZE, IOU_THRESHOLD, MAX_OBJECT_CNT,
                    PERSON_CONF, BAG_CONF, XMEM_CONFIG, YOLO_EVERY)
from inference.inference_utils import (add_new_classes_to_dict,
                                       generate_colors_dict,
                                       get_iou_filtered_yolo_mask_bboxes,
                                       merge_masks, overlay_mask_on_image)
from inference.interact.interactive_utils import torch_prob_to_numpy_mask
from tracker import Tracker
from pose_estimation import Yolov8PoseModel

def preprocess_video(input_video_path, output_video_path, target_width, target_height):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        required=True, help='Path to input video')
    parser.add_argument(
        '--width', type=int, default=INFERENCE_SIZE[0], required=False, help='Inference width')
    parser.add_argument(
        '--height', type=int, default=INFERENCE_SIZE[1], required=False, help='Inference height')
    parser.add_argument('--frames_to_propagate', type=int,
                        default=None, required=False, help='Frames to propagate')
    parser.add_argument('--output_video_path', type=str, default=None,
                        required=False, help='Output video path to save')
    parser.add_argument('--device', type=str, default=DEVICE,
                        required=False, help='GPU id')
    parser.add_argument('--person_conf', type=float, default=PERSON_CONF,
                        required=False, help='YOLO person confidence')
    parser.add_argument('--bag_conf', type=float, default=BAG_CONF,
                        required=False, help='YOLO bag confidence')
    parser.add_argument('--iou_thresh', type=float, default=IOU_THRESHOLD,
                        required=False, help='IOU threshold to find new persons bboxes')
    parser.add_argument('--yolo_every', type=int, default=YOLO_EVERY,
                        required=False, help='Find new persons with YOLO every N frames')
    parser.add_argument('--output_path', type=str,
                        default='tracking_results.csv', required=False, help='Output filepath')
    parser.add_argument('--preprocessed_video_path', type=str,
                        default='preprocessed_video.mp4', required=False, help='Path to save preprocessed video')

    args = parser.parse_args()

    # 비디오 전처리: 해상도 낮추기
    preprocess_video(args.video_path, args.preprocessed_video_path, args.width, args.height)

    if torch.cuda.device_count() > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    cap = cv2.VideoCapture(args.preprocessed_video_path)
    df = pd.DataFrame(
        columns=['frame_id', 'object_type', 'object_id', 'x1', 'y1', 'x2', 'y2'])

    if args.output_video_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), fps, (args.width, args.height))

    yolov8pose_model = Yolov8PoseModel(DEVICE, PERSON_CONF, BAG_CONF)
    tracker = Tracker(XMEM_CONFIG, MAX_OBJECT_CNT, DEVICE)
    objects_in_video = False

    class_color_mapping = generate_colors_dict(MAX_OBJECT_CNT + 1)

    current_frame_index = 0

    # 사람과 가방에 대한 라벨 맵핑을 별도로 관리
    person_label_mapping = {}
    bag_label_mapping = {}

    with torch.cuda.amp.autocast(enabled=True):

        while (cap.isOpened()):
            _, frame = cap.read()

            if frame is None or (args.frames_to_propagate is not None and current_frame_index == args.frames_to_propagate):
                break

            if current_frame_index % args.yolo_every == 0:
                person_bboxes, bag_bboxes = yolov8pose_model.get_filtered_bboxes_by_confidence(frame)

            if len(person_bboxes) > 0 or len(bag_bboxes) > 0:
                objects_in_video = True
            else:
                masks = []
                mask_bboxes_with_idx = []

            if objects_in_video:
                all_bboxes = person_bboxes + bag_bboxes
                if len(person_label_mapping) == 0 and len(bag_label_mapping) == 0:  # First objects in video
                    # 사람 마스크 생성 및 라벨 맵핑
                    person_mask = tracker.create_mask_from_img(frame, person_bboxes, device='0')
                    unique_person_labels = np.unique(person_mask)
                    person_label_mapping = {label: idx for idx, label in enumerate(unique_person_labels)}
                    person_mask = np.array([person_label_mapping[label] for label in person_mask.flat]).reshape(person_mask.shape)
                    
                    # 가방 마스크 생성 및 라벨 맵핑
                    bag_mask = tracker.create_mask_from_img(frame, bag_bboxes, device='0')
                    unique_bag_labels = np.unique(bag_mask)
                    bag_label_mapping = {label: idx + len(person_label_mapping) for idx, label in enumerate(unique_bag_labels)}
                    bag_mask = np.array([bag_label_mapping[label] for label in bag_mask.flat]).reshape(bag_mask.shape)

                    # 마스크 병합
                    mask = merge_masks(person_mask, bag_mask)
                    prediction = tracker.add_mask(frame, mask)
                elif len(filtered_bboxes) > 0:  # Additional/new objects in video
                    # 사람 마스크 업데이트 및 라벨 맵핑
                    person_mask = tracker.create_mask_from_img(frame, [bbox for bbox in filtered_bboxes if bbox in person_bboxes], device='0')
                    unique_person_labels = np.unique(person_mask)
                    person_label_mapping = add_new_classes_to_dict(unique_person_labels, person_label_mapping)
                    person_mask = np.array([person_label_mapping[label] for label in person_mask.flat]).reshape(person_mask.shape)
                    
                    # 가방 마스크 업데이트 및 라벨 맵핑
                    bag_mask = tracker.create_mask_from_img(frame, [bbox for bbox in filtered_bboxes if bbox in bag_bboxes], device='0')
                    unique_bag_labels = np.unique(bag_mask)
                    bag_label_mapping = add_new_classes_to_dict(unique_bag_labels, bag_label_mapping)
                    bag_mask = np.array([bag_label_mapping[label] for label in bag_mask.flat]).reshape(bag_mask.shape)

                    # 마스크 병합
                    mask = merge_masks(person_mask, bag_mask)
                    merged_mask = merge_masks(masks.squeeze(0), torch.tensor(mask))
                    prediction = tracker.add_mask(frame, merged_mask.squeeze(0).numpy())
                    filtered_bboxes = []
                else:  # Only predict
                    prediction = tracker.predict(frame)

                masks = torch.tensor(torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
                mask_bboxes_with_idx = tracker.masks_to_boxes_with_ids(masks)

                if current_frame_index % args.yolo_every == 0:
                    filtered_bboxes = get_iou_filtered_yolo_mask_bboxes(
                        all_bboxes, mask_bboxes_with_idx, iou_threshold=args.iou_thresh)

            # VISUALIZATION
            if args.output_video_path is not None:
                if len(mask_bboxes_with_idx) > 0:
                    for bbox in mask_bboxes_with_idx:
                        cv2.rectangle(frame, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), (255, 255, 0), 2)
                        cv2.putText(frame, f'{bbox[0]}', (int(bbox[1])-10, int(bbox[2])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    visualization = overlay_mask_on_image(frame, masks, class_color_mapping, alpha=0.75)
                    visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
                    result.write(visualization)
                else:
                    result.write(frame)

            if len(mask_bboxes_with_idx) > 0:
                for bbox in mask_bboxes_with_idx:
                    object_id = bbox[0]
                    x1 = bbox[1]
                    y1 = bbox[2]
                    x2 = bbox[3]
                    y2 = bbox[4]
                    object_type = 'person' if object_id in person_label_mapping.values() else 'bag'
                    df.loc[len(df.index)] = [int(current_frame_index), object_type, object_id, x1, y1, x2, y2]
            else:
                df.loc[len(df.index)] = [int(current_frame_index), None, None, None, None, None, None]

            print(f'current_frame_index: {current_frame_index}, objects in frame: {len(mask_bboxes_with_idx)}')
            current_frame_index += 1

    df.to_csv(args.output_path, index=False)
    if args.output_video_path is not None:
        result.release()

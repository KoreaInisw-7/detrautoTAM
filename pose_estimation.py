import os
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Yolov8PoseModel:
    def __init__(self, device: str, person_conf, bag_conf):
        self.device = f"cuda:{device}" if isinstance(device, int) or device.isdigit() else device
        self.person_conf = person_conf
        self.bag_conf = bag_conf
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50').to(self.device)
        self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        self.person_categories = ['person']
        self.bag_categories = ['backpack', 'handbag', 'suitcase']
        
    def run_inference(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        return results
    
    def get_filtered_bboxes_by_confidence(self, image):
        results = self.run_inference(image)
        
        person_bboxes = []
        bag_bboxes = []
        
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            label_name = self.model.config.id2label[label.item()]
            if score > self.person_conf and label_name in self.person_categories:
                person_bboxes.append(box.int().tolist())
            elif score > self.bag_conf and label_name in self.bag_categories:
                bag_bboxes.append(box.int().tolist())
        
        return person_bboxes, bag_bboxes
    
    def get_filtered_bboxes_by_size(self, bboxes, image, percentage=10):
        image_size = image.shape[:2]
        min_bbox_width = image_size[1] * (percentage/100)  # width
        min_bbox_height = image_size[0] * (percentage/100)  # height

        filtered_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width >= min_bbox_width and bbox_height >= min_bbox_height:
                filtered_bboxes.append(bbox)

        return filtered_bboxes

    def visualize(self, frame, results, categories):
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        ax = plt.gca()

        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            label_name = self.model.config.id2label[label.item()]
            if score > 0.5 and label_name in categories:
                box = box.detach().cpu().numpy()
                x, y, w, h = box
                rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(x, y, f'{label_name}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.show()

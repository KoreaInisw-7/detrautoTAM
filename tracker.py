# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry

from skimage import measure

from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import (image_to_torch,
                                                  index_numpy_to_one_hot_torch)
from model.network import XMem

class Tracker:
    def __init__(self, xmem_config, max_obj_cnt, device):
        self.device = device
        self.xmem_config = xmem_config
        self.max_obj_cnt = max_obj_cnt
        if self.device.lower() != 'cpu':
            self.network = XMem(
                self.xmem_config, './saves/XMem.pth').eval().to('cuda')
        else:
            self.network = XMem(
                self.xmem_config, './saves/XMem.pth', map_location='cpu').eval()
        self.processor = InferenceCore(self.network, config=self.xmem_config)
        self.processor.set_all_labels(range(1, self.max_obj_cnt+1))

    def masks_on_im(self, masks, image):
        result = np.zeros_like(image, dtype=np.uint8)

        for mask in masks:
            color = np.random.randint(0, 256, size=3)

            colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            masked_region = colored_mask * color
            result += masked_region.astype(np.uint8)

        return result

    def create_mask_from_img(self, image, person_bboxes, bag_bboxes, sam_checkpoint='./saves/mobile_sam.pt', model_type='vit_t', device='0'):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        if self.device.lower() != 'cpu':
            sam.to(device=f'cuda:{device}')
        else:
            sam.to(device='cpu')
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        def generate_masks(bboxes):
            input_boxes = torch.tensor(bboxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

            masks = []
            for box in transformed_boxes:
                mask, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=box.unsqueeze(0),
                    multimask_output=False,
                )
                values, counts = torch.unique(mask, return_counts=True)
                value_count = [(v.item(), c.item()) for v, c in zip(values, counts)]
                value_count = sorted(value_count, key=lambda x: x[1], reverse=True)
                mask[mask != 0] = value_count[0][0] if value_count[0][0] != 0 else value_count[1][0]
                masks.append(mask)
            return masks

        person_masks = generate_masks(person_bboxes)
        bag_masks = generate_masks(bag_bboxes)

        # Combine masks and create a result image
        all_masks = person_masks + bag_masks
        result = self.masks_on_im([mask.cpu().squeeze().numpy().astype(np.uint8) for mask in all_masks], image)
        result = result[:, :, 0]

        # Filter result from small segmented areas
        if len(np.unique(result)) > len(person_bboxes) + len(bag_bboxes) + 1:
            filtered_result_values = []
            mask_uniq_values = torch.unique(torch.tensor(result), return_counts=True)[0].tolist()
            class_pixel_cnts = torch.unique(torch.tensor(result), return_counts=True)[1].tolist()
            sorted_indices = np.argsort(class_pixel_cnts)[::-1].tolist()

            for index in sorted_indices:
                filtered_result_values.append(mask_uniq_values[index])
                if len(filtered_result_values) == len(person_bboxes) + len(bag_bboxes) + 1:
                    break

            for pixel_val in mask_uniq_values:
                if pixel_val not in filtered_result_values:
                    result[result == pixel_val] = 0

        return result, person_masks, bag_masks

    def masks_to_boxes_with_ids(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        unique_values = torch.unique(mask_tensor[mask_tensor != 0])

        bbox_list = []

        for unique_value in unique_values:
            binary_mask = (mask_tensor == unique_value).byte()

            nonzero_coords = torch.nonzero(binary_mask, as_tuple=False)

            if nonzero_coords.numel() > 0:
                min_x = torch.min(nonzero_coords[:, 2])
                min_y = torch.min(nonzero_coords[:, 1])
                max_x = torch.max(nonzero_coords[:, 2])
                max_y = torch.max(nonzero_coords[:, 1])

                bbox = [unique_value.item(), min_x.item(), min_y.item(),
                        max_x.item(), max_y.item()]
                bbox_list.append(bbox)

        return bbox_list

    def predict(self, image):
        if self.device.lower() != 'cpu':
            frame_torch, _ = image_to_torch(image, device='cuda')
        else:
            frame_torch, _ = image_to_torch(image, device='cpu')

        return self.processor.step(frame_torch)

    def add_mask(self, image, mask):
        if self.device.lower() != 'cpu':
            frame_torch, _ = image_to_torch(image, device='cuda')
            mask_torch = index_numpy_to_one_hot_torch(
                mask, self.max_obj_cnt + 1).to('cuda')
        else:
            frame_torch, _ = image_to_torch(image, device='cpu')
            mask_torch = index_numpy_to_one_hot_torch(
                mask, self.max_obj_cnt + 1).to('cpu')
        print('Added new mask')

        return self.processor.step(frame_torch, mask_torch[1:])
    
    def keep_largest_connected_components(self, mask):
        mask_np = mask.squeeze().cpu().numpy()

        unique_values = np.unique(mask_np)
        unique_values = unique_values[unique_values != 0]

        new_mask = np.zeros_like(mask_np)
        
        for class_value in unique_values:
            binary_mask = (mask_np == class_value).astype(np.uint8)
            
            # Dynamic kernel size = (25% of object width, 25% of object height)
            _, _, w, h = cv2.boundingRect(binary_mask)
            kernel = (max(1, int(w // 4)), max(1, int(h // 4)))
            
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            labeled_mask, num_components = measure.label(binary_mask, background=0, return_num=True)
            component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            largest_component = np.argmax(component_sizes) + 1  # +1 because labels start from 1
            new_mask[labeled_mask == largest_component] = class_value

        new_mask = torch.from_numpy(new_mask).unsqueeze(0)
        return new_mask

import math
import re
import ast
from typing import Dict
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from gui_actor.constants import (
    IGNORE_INDEX,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    ACTION_PATTENS_XY
)

def reformat_coordinates(text):
    """
    (1) Find all the coordinates in the text.
    (2) Replace the coordinates with the special tokens.
    (3) Return the new text and the coordinates as a list of (x, y), where x in [0, 1] and y in [0, 1].
    """
    epsilon = 0.001
    def adjust_coord(c):
        """
        Adjust coordinate if it is too close to 0 or 1.
        """
        if abs(c) < epsilon:
            return epsilon
        elif abs(c - 1) < epsilon:
            return 1 - epsilon
        return c

    all_matches = []
    for pattern in ACTION_PATTENS_XY:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            all_matches.append((match.start(), match.groups()))
        if pattern == ACTION_PATTENS_XY[0]:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        else:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}, {DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        text = re.sub(
            pattern,
            target_text,
            text
        )
    
    coordinates = []
    all_matches.sort(key=lambda x: x[0])
    # Extract coordinates in order
    for _, groups in all_matches:
        # When two coordinate values are found, parse them as one (x, y) pair.
        if len(groups) == 2:
            x_str, y_str = groups
            x = adjust_coord(ast.literal_eval(x_str))
            y = adjust_coord(ast.literal_eval(y_str))
            coordinates.append((x, y))
        # When four coordinate values are found, parse them as two pairs.
        elif len(groups) == 4:
            x1_str, y1_str, x2_str, y2_str = groups
            x1 = adjust_coord(ast.literal_eval(x1_str))
            y1 = adjust_coord(ast.literal_eval(y1_str))
            x2 = adjust_coord(ast.literal_eval(x2_str))
            y2 = adjust_coord(ast.literal_eval(y2_str))
            coordinates.append((x1, y1))
            coordinates.append((x2, y2))
    
    return text, coordinates

def get_token_index(image_processor, image, point_x, point_y):
    """
    Get the index of the visual token that contains the point (x, y).
    Args:
        image_processor: the image processor
        image: the image in PIL format
        point_x: the x coordinate of the point, in [0, 1].
        point_y: the y coordinate of the point, in [0, 1].
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")
    
    # get the original image size and the resized image size
    image = image[0]
    w, h = image.size
    px, py = w * point_x, h * point_y
    # rank0_print(f"px: {px}, py: {py}")
    # get the token index
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = math.floor(px / merge_patch_size)
    y_index = math.floor(py / merge_patch_size)
    
    visual_token_index = y_index * (w // merge_patch_size) + x_index

    # merge all above print into one line
    return visual_token_index

def get_multi_patch_labels(image_processor, image, bbox_gt):
    """
    Get the multi-patch labels for the bounding box.
    Args:
        image_processor: the image processor
        image: the image in PIL format
        bbox_gt: the bounding box in the format of (x_min, y_min, x_max, y_max) [0,1]
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")

    # Get the original image size and the resized image size
    image = image[0]
    w, h = image.size

    bbox_gt = [bbox_gt[0]*w, bbox_gt[1]*h, bbox_gt[2]*w, bbox_gt[3]*h]
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox_gt
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    assert w % merge_patch_size == 0 and h % merge_patch_size == 0, f"Image size {w}x{h} is not divisible by merge_patch_size {merge_patch_size}"
    grid_h, grid_w = h // merge_patch_size, w // merge_patch_size

    binary_mask = torch.zeros(grid_h * grid_w)
    # Iterate through all patches, check if they overlap with the bounding box
    for y_idx in range(grid_h):
        for x_idx in range(grid_w):
            # Calculate patch boundaries
            patch_x_min = x_idx * merge_patch_size
            patch_y_min = y_idx * merge_patch_size
            patch_x_max = patch_x_min + merge_patch_size
            patch_y_max = patch_y_min + merge_patch_size
            
            # Check if patch overlaps with the bounding box
            if not (patch_x_max <= x_min or patch_x_min >= x_max or 
                    patch_y_max <= y_min or patch_y_min >= y_max):
                # Calculate patch index in the flattened grid
                patch_idx = y_idx * grid_w + x_idx
                binary_mask[patch_idx] = 1

    return binary_mask

def token_index_to_coordinates(image_processor, visual_token_index, image_width, image_height):
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = visual_token_index % (image_width // merge_patch_size)
    y_index = visual_token_index // (image_width // merge_patch_size)
    px = x_index * merge_patch_size + merge_patch_size / 2
    py = y_index * merge_patch_size + merge_patch_size / 2
    return px, py

class GUIActorFiftyOneCollator:
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
        self.pointer_pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)
        
    def __call__(self, batch):
        # Process each sample
        processed_samples = []
        for item in batch:
            messages = item['message_payloads'][0]  # Get first annotation
            filepath = item['filepath']
            
            # Load image
            image = Image.open(filepath).convert('RGB')
            
            # Extract point_gt or bbox_gt from assistant message
            assistant_msg = messages[-1]  # Last message is assistant
            point_gt = assistant_msg.get('point_gt')
            bbox_gt = assistant_msg.get('bbox_gt')
            
            # Process the conversation with image
            messages_with_image = []
            for msg in messages:
                if msg['role'] == 'user':
                    # Replace filepath with actual image for processing
                    content = []
                    for c in msg['content']:
                        if c['type'] == 'image':
                            content.append({'type': 'image', 'image': image})
                        else:
                            content.append(c)
                    messages_with_image.append({'role': msg['role'], 'content': content})
                else:
                    messages_with_image.append(msg)
            
            # Process vision info to get resized images
            image_inputs, _ = process_vision_info(messages_with_image)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages_with_image,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Replace coordinates with special tokens and extract them
            text, coordinates = reformat_coordinates(text)
            
            # Process through processor
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt"
            )
            
            # Compute visual token indices and multi-patch labels
            visual_token_indices = []
            multi_patch_labels = []
            
            if coordinates and image_inputs:
                for coord in coordinates:
                    x, y = coord
                    visual_idx = get_token_index(
                        self.processor.image_processor,
                        image_inputs,
                        x, y
                    )
                    visual_token_indices.append(visual_idx)
                    
                    # Add multi-patch labels for bbox
                    if bbox_gt:
                        patch_mask = get_multi_patch_labels(
                            self.processor.image_processor,
                            image_inputs,
                            bbox_gt
                        )
                        multi_patch_labels.append(patch_mask)
                    elif point_gt:
                        # For points, create small region mask
                        n_visual = (inputs['image_grid_thw'][0][0] * 
                                  inputs['image_grid_thw'][0][1] // 
                                  (self.processor.image_processor.merge_size ** 2))
                        patch_mask = torch.zeros(n_visual)
                        patch_mask[visual_idx] = 1.0
                        multi_patch_labels.append(patch_mask)
            
            # Create labels (shift input_ids for language modeling)
            labels = inputs['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
            
            sample_dict = {
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'labels': labels[0],
            }
            
            if image_inputs:
                sample_dict['pixel_values'] = inputs['pixel_values'][0]
                sample_dict['image_grid_thw'] = inputs['image_grid_thw'][0]
            
            if visual_token_indices:
                sample_dict['visual_token_indices_of_coordinates'] = torch.tensor(visual_token_indices)
                sample_dict['multi_patch_labels'] = torch.stack(multi_patch_labels) if multi_patch_labels else None
            
            processed_samples.append(sample_dict)
        
        # Batch the samples
        return self.collate_batch(processed_samples)
    
    def collate_batch(self, samples):
        batch = {}
        
        # Stack tensors
        batch['input_ids'] = torch.stack([s['input_ids'] for s in samples])
        batch['attention_mask'] = torch.stack([s['attention_mask'] for s in samples])
        batch['labels'] = torch.stack([s['labels'] for s in samples])
        
        # Handle images
        if 'pixel_values' in samples[0]:
            batch['pixel_values'] = torch.cat([s['pixel_values'].unsqueeze(0) for s in samples])
            batch['image_grid_thw'] = torch.stack([s['image_grid_thw'] for s in samples])
        
        # Handle coordinate supervision (list of tensors, one per sample)
        if 'visual_token_indices_of_coordinates' in samples[0]:
            batch['visual_token_indices_of_coordinates'] = [s.get('visual_token_indices_of_coordinates') for s in samples]
            batch['multi_patch_labels'] = [s.get('multi_patch_labels') for s in samples]
        
        return batch
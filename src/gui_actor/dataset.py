import math
import re
import ast
from typing import Dict
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from gui_actor.constants import (
    IGNORE_INDEX,  # Token ID to ignore in loss calculation
    DEFAULT_POINTER_START_TOKEN,  # Special token marking start of pointer sequence
    DEFAULT_POINTER_PAD_TOKEN,  # Special token for padding pointer sequences
    DEFAULT_POINTER_END_TOKEN,  # Special token marking end of pointer sequence
    ACTION_PATTENS_XY  # Regex patterns for extracting coordinate pairs
)

def reformat_coordinates(text):
    """
    Process text to extract and standardize coordinate references, replacing them with special tokens.
    
    This function:
    1. Finds all coordinate patterns in the text (both single points and coordinate pairs)
    2. Extracts and adjusts coordinates to be within valid bounds
    3. Replaces coordinate patterns with special pointer tokens
    
    Args:
        text (str): Input text containing coordinate references
        
    Returns:
        tuple: (processed_text, coordinates) where:
            - processed_text (str): Text with coordinates replaced by special tokens
            - coordinates (list): List of extracted (x,y) coordinate tuples
    """
    # Small value to ensure coordinates stay within [0,1] bounds
    epsilon = 0.001
    
    def adjust_coord(c):
        """Adjust coordinate to avoid exact 0 or 1 values for numerical stability"""
        if abs(c) < epsilon:
            return epsilon  # Avoid exact 0
        elif abs(c - 1) < epsilon:
            return 1 - epsilon  # Avoid exact 1
        return c

    # Store all coordinate pattern matches with their positions for ordered processing
    all_matches = []
    for pattern in ACTION_PATTENS_XY:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            all_matches.append((match.start(), match.groups()))
    
    # Sort matches by position to maintain order
    all_matches.sort(key=lambda x: x[0])
    
    # Extract and process coordinates from matches
    coordinates = []
    for _, groups in all_matches:
        if len(groups) == 2:  # Single point pattern (x,y)
            x = adjust_coord(float(groups[0]))
            y = adjust_coord(float(groups[1]))
            coordinates.append((x, y))
        elif len(groups) == 4:  # Two point pattern (x1,y1,x2,y2)
            x1 = adjust_coord(float(groups[0]))
            y1 = adjust_coord(float(groups[1]))
            x2 = adjust_coord(float(groups[2]))
            y2 = adjust_coord(float(groups[3]))
            coordinates.append((x1, y1))
            coordinates.append((x2, y2))
    
    # Replace coordinate patterns with special pointer token sequences
    for pattern in ACTION_PATTENS_XY:
        if pattern == ACTION_PATTENS_XY[0]:  # Single point pattern
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        else:  # Two point pattern
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}, {DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        text = re.sub(pattern, target_text, text)
    
    return text, coordinates

def get_token_index(image_processor, image, point_x, point_y):
    """
    Convert a point coordinate to its corresponding visual token index.
    
    Args:
        image_processor: Processor containing patch size configuration
        image: List containing single PIL image
        point_x: X coordinate normalized to [0,1]
        point_y: Y coordinate normalized to [0,1]
        
    Returns:
        int: Index of the visual token containing the point
        
    Raises:
        ValueError: If more than one image provided
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")
    
    # Convert normalized coordinates to pixel coordinates
    image = image[0]
    w, h = image.size
    px, py = w * point_x, h * point_y
    
    # Calculate token grid position
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = math.floor(px / merge_patch_size)
    y_index = math.floor(py / merge_patch_size)
    
    # Convert 2D grid position to flattened index
    visual_token_index = y_index * (w // merge_patch_size) + x_index

    return visual_token_index

def get_multi_patch_labels(image_processor, image, bbox_gt):
    """
    Generate binary mask indicating which visual tokens overlap with a bounding box.
    
    Args:
        image_processor: Processor containing patch size configuration
        image: List containing single PIL image
        bbox_gt: Bounding box coordinates [x_min, y_min, x_max, y_max] normalized to [0,1]
        
    Returns:
        torch.Tensor: Binary mask of shape (H*W,) where H,W are grid dimensions
        
    Raises:
        ValueError: If more than one image provided
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")

    # Convert normalized bbox to pixel coordinates
    image = image[0]
    w, h = image.size
    bbox_gt = [bbox_gt[0]*w, bbox_gt[1]*h, bbox_gt[2]*w, bbox_gt[3]*h]
    x_min, y_min, x_max, y_max = bbox_gt
    
    # Clamp coordinates to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Calculate grid dimensions
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    assert w % merge_patch_size == 0 and h % merge_patch_size == 0, f"Image size {w}x{h} is not divisible by merge_patch_size {merge_patch_size}"
    grid_h, grid_w = h // merge_patch_size, w // merge_patch_size

    # Create binary mask for overlapping patches
    binary_mask = torch.zeros(grid_h * grid_w)
    for y_idx in range(grid_h):
        for x_idx in range(grid_w):
            # Get patch boundaries
            patch_x_min = x_idx * merge_patch_size
            patch_y_min = y_idx * merge_patch_size
            patch_x_max = patch_x_min + merge_patch_size
            patch_y_max = patch_y_min + merge_patch_size
            
            # Check for overlap with bbox
            if not (patch_x_max <= x_min or patch_x_min >= x_max or 
                    patch_y_max <= y_min or patch_y_min >= y_max):
                patch_idx = y_idx * grid_w + x_idx
                binary_mask[patch_idx] = 1

    return binary_mask

def token_index_to_coordinates(image_processor, visual_token_index, image_width, image_height):
    """
    Convert a visual token index back to pixel coordinates of its center.
    
    Args:
        image_processor: Processor containing patch size configuration
        visual_token_index: Index of visual token in flattened grid
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        
    Returns:
        tuple: (px, py) center coordinates of the token in pixels
    """
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = visual_token_index % (image_width // merge_patch_size)
    y_index = visual_token_index // (image_width // merge_patch_size)
    px = x_index * merge_patch_size + merge_patch_size / 2
    py = y_index * merge_patch_size + merge_patch_size / 2
    return px, py

class GUIActorFiftyOneCollator:
    """
    Collator class for processing GUI interaction data from FiftyOne dataset.
    
    This class handles:
    - Loading and preprocessing images
    - Extracting coordinate information
    - Tokenizing text
    - Computing visual token indices and patch labels
    - Batching samples together
    """
    
    def __init__(self, processor, tokenizer):
        """
        Initialize collator with processors.
        
        Args:
            processor: Vision-language processor for multimodal inputs
            tokenizer: Tokenizer for text processing
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.pointer_pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)
        
    def __call__(self, batch):
        """
        Process a batch of samples from the dataset.
        
        Args:
            batch: List of samples, each containing message_payloads and filepath
            
        Returns:
            dict: Processed batch with input_ids, attention_mask, labels, and visual features
        """
        processed_samples = []
        for item in batch:
            messages = item['message_payloads'][0]  # Already has filepath
            
            # Get ground truth coordinates/bbox from assistant message
            assistant_msg = messages[-1]
            point_gt = assistant_msg.get('point_gt')
            bbox_gt = assistant_msg.get('bbox_gt')
            
            # Process images directly from messages with filepath
            image_inputs, _ = process_vision_info(messages)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Extract and standardize coordinates
            text, coordinates = reformat_coordinates(text)
            
            # Process through vision-language processor
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
            )
            
            # Compute visual token indices and patch labels
            visual_token_indices = []
            multi_patch_labels = []
            
            if coordinates and image_inputs:
                for coord in coordinates:
                    x, y = coord
                    # Get token index for coordinate
                    visual_idx = get_token_index(
                        self.processor.image_processor,
                        image_inputs,
                        x, y
                    )
                    visual_token_indices.append(visual_idx)
                    
                    # Generate patch labels
                    if bbox_gt:
                        # For bounding boxes, get overlapping patches
                        patch_mask = get_multi_patch_labels(
                            self.processor.image_processor,
                            image_inputs,
                            bbox_gt
                        )
                        multi_patch_labels.append(patch_mask)
                    elif point_gt:
                        # For points, mark single patch
                        n_visual = (inputs['image_grid_thw'][0][0] * 
                                  inputs['image_grid_thw'][0][1] // 
                                  (self.processor.image_processor.merge_size ** 2))
                        patch_mask = torch.zeros(n_visual)
                        patch_mask[visual_idx] = 1.0
                        multi_patch_labels.append(patch_mask)
            
            # Create language modeling labels
            labels = inputs['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
            
            # Combine all features for this sample
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
        
        # Collate individual samples into batch
        return self.collate_batch(processed_samples)
    
    def collate_batch(self, samples):
        """
        Collate individual samples into a batch.
        
        Args:
            samples: List of processed sample dictionaries
            
        Returns:
            dict: Batched samples with stacked tensors
        """
        batch = {}
        
        # Stack common tensors
        batch['input_ids'] = torch.stack([s['input_ids'] for s in samples])
        batch['attention_mask'] = torch.stack([s['attention_mask'] for s in samples])
        batch['labels'] = torch.stack([s['labels'] for s in samples])
        
        # Handle optional image features
        if 'pixel_values' in samples[0]:
            batch['pixel_values'] = torch.cat([s['pixel_values'].unsqueeze(0) for s in samples])
            batch['image_grid_thw'] = torch.stack([s['image_grid_thw'] for s in samples])
        
        # Handle optional coordinate features
        if 'visual_token_indices_of_coordinates' in samples[0]:
            batch['visual_token_indices_of_coordinates'] = [s.get('visual_token_indices_of_coordinates') for s in samples]
            batch['multi_patch_labels'] = [s.get('multi_patch_labels') for s in samples]
        
        return batch
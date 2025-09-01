import argparse
import os
import sys
import json
import fiftyone as fo
from fiftyone.utils.torch import GetItem
import fiftyone.utils.random as four


KP_SYSTEM_MESSAGE = """You are a GUI Agent specialized in interacting with the FiftyOne application. Given a screenshot of the current FiftyOne GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction.

You should output a response indicating the element type to interact with, action to be taken on that element, correct position of the action, and any additional metadata. 

Your response must be a valid JSON wrapped exactly this format:

```json
{{"element_info": {element_info}, "label": "{label}", "points": {points}, "custom_metadata": {custom_metadata}}}
```"""

BB_SYSTEM_MESSAGE = """You are a GUI Agent specialized in interacting with the FiftyOne application. Given a screenshot of the current FiftyOne GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction.

You should output a response indicating the element type to interact with, action to be taken on that element, correct position of the action, and any additional metadata. 

Your response must be a valid JSON wrapped exactly this format:

```json
{{"element_info": {element_info}, "label": "{label}", "bounding_box": {bounding_box}, "custom_metadata": {custom_metadata}}}
```"""

def add_message_payload_to_dataset(dataset):
    """
    Add message payloads to FiftyOne dataset annotations for GUI-Actor training.
    
    This function processes both keypoint (single-point) and detection (bounding box)
    annotations, converting them into conversation format for vision-language model training.
    Each annotation gets a complete conversation with system prompt, user query, and 
    assistant response containing both natural language and structured JSON.
    
    Args:
        dataset: FiftyOne dataset with 'keypoints' and/or 'detections' fields
        
    Modifies:
        Adds 'message_payload' field to each annotation containing the formatted
        conversation for training
    """
    
    for sample in dataset.iter_samples(autosave=True, progress=True):
        filepath = sample["filepath"]
        
        # Process keypoints (point-based interactions like clicks)
        if sample.keypoints:
            for kp in sample.keypoints.keypoints:
                # Extract keypoint attributes with safe defaults
                task_desc = getattr(kp, 'task_description', '')
                element_info = getattr(kp, 'element_info', None)
                action = getattr(kp, 'label', 'click')  # Default action is click
                points = getattr(kp, 'points', [])
                custom_metadata = getattr(kp, 'custom_metadata', None)
                
                # Ensure element_info is a proper dict or string
                if element_info is None or element_info == {} or element_info == '':
                    element_info = "ui_element"  # Simple string fallback
                elif isinstance(element_info, dict) and not element_info:
                    element_info = "ui_element"  # Empty dict -> string
                elif not isinstance(element_info, (dict, str)):
                    element_info = str(element_info)  # Convert to string if weird type
                
                # Ensure custom_metadata is a proper dict
                if custom_metadata is None or custom_metadata == {} or custom_metadata == '':
                    custom_metadata = {"type": "point_interaction"}
                elif not isinstance(custom_metadata, dict):
                    custom_metadata = {"value": str(custom_metadata)}
                
                # Only process if we have valid coordinates
                if points and len(points) > 0:
                    x, y = points[0]
                    
                    # Build the JSON response object
                    json_response = {
                        "action": action,
                        "element_info": element_info,
                        "points": [[round(x, 4), round(y, 4)]],  # Round for cleaner output
                        "custom_metadata": custom_metadata
                    }
                    
                    # Create natural language response with embedded JSON
                    # Note: coordinates in text match those used for pointer tokens
                    response_text = f"""I can {action} the {element_info if isinstance(element_info, str) else 'element'} at x={round(x, 4)}, y={round(y, 4)} to complete this task. Here is the valid JSON response: ```json {json.dumps(json_response)}```"""
                    
                    # Create the full conversation format
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": KP_SYSTEM_MESSAGE}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": filepath},
                                {"type": "text", "text": task_desc}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response_text}
                            ],
                            "recipient": "os",
                            "end_turn": True,
                            "point_gt": points[0] if points else None  # Ground truth for pointer loss
                        }
                    ]
                    
                    kp.message_payload = messages
        
        # Process detections (bounding box interactions like drag, select)
        if sample.detections:
            for det in sample.detections.detections:
                # Extract detection attributes with safe defaults
                task_desc = getattr(det, 'task_description', '')
                element_info = getattr(det, 'element_info', None)
                action = getattr(det, 'label', 'select')  # Default action is select
                bounding_box = getattr(det, 'bounding_box', [])
                custom_metadata = getattr(det, 'custom_metadata', None)
                
                # Ensure element_info is a proper dict or string
                if element_info is None or element_info == {} or element_info == '':
                    element_info = "ui_region"  # Simple string fallback
                elif isinstance(element_info, dict) and not element_info:
                    element_info = "ui_region"  # Empty dict -> string
                elif not isinstance(element_info, (dict, str)):
                    element_info = str(element_info)  # Convert to string if weird type
                
                # Ensure custom_metadata is a proper dict
                if custom_metadata is None or custom_metadata == {} or custom_metadata == '':
                    custom_metadata = {"type": "bbox_interaction"}
                elif not isinstance(custom_metadata, dict):
                    custom_metadata = {"value": str(custom_metadata)}
                
                # Only process if we have valid bounding box
                if bounding_box and len(bounding_box) == 4:
                    # FiftyOne format: [x, y, width, height] in relative coords [0,1]
                    x, y, width, height = bounding_box
                    
                    # Convert to [x_min, y_min, x_max, y_max] for consistency
                    x_min = round(x, 4)
                    y_min = round(y, 4)
                    x_max = round(x + width, 4)
                    y_max = round(y + height, 4)
                    bbox_gt_format = [x_min, y_min, x_max, y_max]
                    
                    # Build the JSON response object
                    json_response = {
                        "action": action,
                        "element_info": element_info,
                        "bounding_box": bbox_gt_format,
                        "custom_metadata": custom_metadata
                    }
                    
                    # Create natural language response with embedded JSON
                    # Note: from_coord/to_coord format matches training patterns
                    response_text = f"""I can {action} the {element_info if isinstance(element_info, str) else 'region'} from_coord=[{x_min}, {y_min}] to_coord=[{x_max}, {y_max}] to complete this task. Here is the valid JSON response: ```json {json.dumps(json_response)}```"""
                    
                    # Create the full conversation format
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": BB_SYSTEM_MESSAGE}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": filepath},
                                {"type": "text", "text": task_desc}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response_text}
                            ],
                            "recipient": "os",
                            "end_turn": True,
                            "bbox_gt": bbox_gt_format  # Ground truth for pointer loss
                        }
                    ]
                    
                    det.message_payload = messages

class DataGetter(GetItem):
    @property
    def required_keys(self):
        return ['filepath', 'keypoints', 'detections']

    def __call__(self, d):
        message_payloads = []
        
        # Extract message_payload from all keypoints in the sample
        keypoints = d.get("keypoints")
        if keypoints is not None and hasattr(keypoints, 'keypoints'):
            for keypoint in keypoints.keypoints:
                if hasattr(keypoint, 'message_payload') and keypoint.message_payload is not None:
                    message_payloads.append(keypoint.message_payload)
        
        # Extract message_payload from all detections in the sample
        detections = d.get("detections")
        if detections is not None and hasattr(detections, 'detections'):
            for detection in detections.detections:
                if hasattr(detection, 'message_payload') and detection.message_payload is not None:
                    message_payloads.append(detection.message_payload)
        
        return {
            "filepath": d.get("filepath", ""),
            "message_payload": message_payloads,
        }


class FlattenedDataset:
    """
    Flattens a FiftyOne torch dataset so each item is a single message_payload
    with its associated filepath.
    """
    def __init__(self, fiftyone_torch_dataset):
        self.items = []
        for sample in fiftyone_torch_dataset:
            filepath = sample["filepath"]
            for message_payload in sample["message_payload"]:
                if message_payload:  # Only add non-empty payloads
                    self.items.append({
                        "filepath": filepath,
                        "message_payload": message_payload
                    })
        print(f"FlattenedDataset created with {len(self.items)} items")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
def create_torch_dataset(dataset_name):
    """
    Create PyTorch datasets from a FiftyOne dataset.
    
    Args:
        dataset_name: Name of the FiftyOne dataset to process
    
    Returns:
        train_dataset, val_dataset: Flattened PyTorch datasets for training and validation
    """
    print(f"Loading FiftyOne dataset: {dataset_name}")
    dataset = fo.load_dataset(dataset_name)
    
    print("Adding message payloads to dataset...")
    add_message_payload_to_dataset(dataset)
    
    print("Performing random split...")
    # Split the dataset into train and validation
    four.random_split(dataset, {"train": 0.8, "val": 0.2})
    
    print("Creating PyTorch datasets...")
    # Create views for train and validation
    train_view = dataset.match_tags("train")
    val_view = dataset.match_tags("val")
    
    # Create torch datasets using DataGetter
    train_torch_dataset = train_view.to_torch(DataGetter())
    val_torch_dataset = val_view.to_torch(DataGetter())
    
    print(f"Intermediate datasets: train={len(train_torch_dataset)}, val={len(val_torch_dataset)}")
    
    # Flatten the datasets so each item is a single message_payload
    print("Flattening datasets...")
    train_dataset = FlattenedDataset(train_torch_dataset)
    val_dataset = FlattenedDataset(val_torch_dataset)
    
    print(f"Final flattened datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Create PyTorch datasets from a FiftyOne dataset")
    parser.add_argument("fiftyone_dataset_name", type=str, help="Name of the FiftyOne dataset to process")
    
    args = parser.parse_args()
    
    train_dataset, val_dataset = create_torch_dataset(args.fiftyone_dataset_name)
    
    print(f"Successfully created PyTorch datasets from {args.fiftyone_dataset_name}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")


if __name__ == "__main__":
    main()
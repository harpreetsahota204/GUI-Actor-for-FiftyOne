import argparse
import os
import sys
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
    Add message payload to all keypoints and detections, where the label
    represents the ACTION to be performed (click, type, select, etc.).
    
    Converts bounding boxes from [x, y, width, height] to [x_min, y_min, x_max, y_max] for bbox_gt.
    """
    
    for sample in dataset.iter_samples(autosave=True, progress=True):
        filepath = sample["filepath"]
        
        # Process keypoints (point-based interactions)
        if sample.keypoints:
            for kp in sample.keypoints.keypoints:
                # Extract keypoint attributes
                task_desc = getattr(kp, 'task_description', '')
                element_info = getattr(kp, 'element_info', {})
                action = getattr(kp, 'label', '')
                points = getattr(kp, 'points', [])
                custom_metadata = getattr(kp, 'custom_metadata', {})
                
                # Extract coordinates for pointer loss
                if points and len(points) > 0:
                    x, y = points[0]
                    
                    # Create response with action and location
                    response_text = f"""I can {action} the {element_info} at x={x}, y={y} to complete this task. Here is the valid JSON response: ```json {{"action": "{action}", "element_info": {element_info}, "points": {points}, "custom_metadata": {custom_metadata}}}```"""
                    
                    # Create message payload in the correct format
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
                            "point_gt": points[0] if points else None
                        }
                    ]
                    
                    kp.message_payload = messages
        
        # Process detections (region-based interactions)
        if sample.detections:
            for det in sample.detections.detections:
                # Extract detection attributes
                task_desc = getattr(det, 'task_description', '')
                element_info = getattr(det, 'element_info', '')
                action = getattr(det, 'label', '')
                bounding_box = getattr(det, 'bounding_box', [])
                custom_metadata = getattr(det, 'custom_metadata', {})
                
                if bounding_box and len(bounding_box) == 4:
                    # Extract bounding box coordinates [x, y, width, height]
                    x, y, width, height = bounding_box
                    
                    # Convert to [x_min, y_min, x_max, y_max] format for bbox_gt
                    x_min = x
                    y_min = y
                    x_max = x + width
                    y_max = y + height
                    bbox_gt_format = [x_min, y_min, x_max, y_max]
                    
                    # Create response with action and bounding box information
                    response_text = f"""I can {action} the {element_info} from_coord=[{x_min}, {y_min}] to_coord=[{x_max}, {y_max}] to complete this task. Here is the valid JSON response: ```json {{"action": "{action}", "element_info": {element_info}, "bounding_box": {bbox_gt_format}, "custom_metadata": {custom_metadata}}}```"""
                    
                    # Create message payload in the correct format
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
                            "bbox_gt": bbox_gt_format
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
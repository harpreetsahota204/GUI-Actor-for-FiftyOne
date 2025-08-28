from PIL import Image, ImageDraw, ImageColor
import json
import os

def dump_args_to_json(model_config, data_processor, model_args, data_args, training_args, output_dir):
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except:
            return False

    save_path = f"{output_dir}/args.json"
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            json.dump({
                "model_config": {k: v for k, v in model_config.__dict__.items() if is_json_serializable(v)},
                "data_processor_config": {k: v for k, v in data_processor.__dict__.items() if is_json_serializable(v)},
                "image_processor_config": {k: v for k, v in data_processor.image_processor.__dict__.items() if is_json_serializable(v)},
                "model_args": {k: v for k, v in model_args.__dict__.items() if is_json_serializable(v)},
                "data_args": {k: v for k, v in data_args.__dict__.items() if is_json_serializable(v)},
                "training_args": {k: v for k, v in training_args.__dict__.items() if is_json_serializable(v)},
            }, f, indent=4)

def do_boxes_overlap(box1, box2):
    """
    Check if two boxes overlap.
    
    Each box is represented as a tuple: (x1, y1, x2, y2)
    Where (x1, y1) is the top-left and (x2, y2) is the bottom-right corner.
    """
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check for no overlap
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True
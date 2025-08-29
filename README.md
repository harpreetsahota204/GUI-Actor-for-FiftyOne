# GUI-Actor Fine-Tuning for FiftyOne Applications

This repository provides a streamlined implementation for fine-tuning Microsoft's GUI-Actor model on custom FiftyOne datasets, enabling vision-language models to interact with specific GUI applications through visual grounding and coordinate prediction.

## Overview

GUI-Actor is a vision-language model designed to understand and interact with graphical user interfaces by predicting precise click coordinates or bounding boxes based on natural language instructions. This implementation adapts the original Microsoft GUI-Actor training pipeline to work with FiftyOne's annotation system, making it accessible for training on custom GUI interaction datasets with realistic hardware constraints.

The key innovation here is bridging FiftyOne's annotation-centric data model with GUI-Actor's conversation-based training format, while implementing memory optimizations that enable training on single GPUs with as little as 24GB of VRAM.

## Key Features

This implementation provides several important capabilities for researchers and developers working with GUI automation:

**Data Pipeline Integration**: Seamless conversion from FiftyOne's keypoint and detection annotations to GUI-Actor's expected conversation format. Each annotation becomes a complete interaction sequence with system prompts, user instructions, and model responses.

**Memory-Optimized Training**: Carefully tuned settings that enable training on consumer GPUs through gradient accumulation, sequence length optimization, and selective parameter unfreezing. The system can train effectively on a single RTX 4090 or similar hardware.

**Flexible Fine-Tuning Strategies**: Support for both conservative fine-tuning (updating only task-specific layers) and aggressive single-application mode (adapting vision and language components for specialized use cases).

**Coordinate Precision Preservation**: Maintains sub-pixel accuracy through careful coordinate normalization and visual token index calculation, ensuring the model learns precise pointing behaviors.

## Installation

Begin by setting up your environment with the required dependencies. This implementation requires Python 3.8 or later and CUDA-capable GPUs.

```bash
# Clone the repository
git clone https://github.com/yourusername/gui-actor-fiftyone.git
cd gui-actor-fiftyone

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision 

# Install required packages
pip install transformers accelerate fiftyone qwen-vl-utils
pip install flash-attn --no-build-isolation  # For Flash Attention 2 support
```

## Preparing Your Dataset

The training pipeline expects GUI interaction data in FiftyOne format with specific annotation structures. Your dataset should contain screenshots with either keypoint annotations (for point clicks) or detection annotations (for region selections).

### Dataset Structure

Each sample in your FiftyOne dataset needs the following components:

```python
import fiftyone as fo

# Create a sample with GUI interaction annotations
sample = fo.Sample(filepath="path/to/screenshot.png")

# For click interactions - use keypoints
keypoint = fo.Keypoint(
    points=[(0.5, 0.3)],  # Normalized coordinates (x, y)
    label="click",  # Action type
    task_description="Click on the Submit button",
    element_info={"type": "button", "text": "Submit"},
    custom_metadata={"confidence": 0.95}
)
sample["keypoints"] = fo.Keypoints(keypoints=[keypoint])

# For drag/select interactions - use detections
detection = fo.Detection(
    bounding_box=[0.1, 0.2, 0.3, 0.4],  # [x, y, width, height] normalized
    label="select",  # Action type
    task_description="Select all text in the editor",
    element_info="text_editor",
    custom_metadata={"element_id": "main-editor"}
)
sample["detections"] = fo.Detections(detections=[detection])
```

The pipeline automatically converts these annotations into the conversation format expected by GUI-Actor, where each interaction becomes a structured dialogue between the user requesting an action and the model responding with coordinates.

### Creating Training and Validation Sets

The system automatically splits your dataset into training and validation sets:

```python
from gui_actor.create_fo_torch_dataset import create_torch_dataset

# Load and prepare your FiftyOne dataset
train_dataset, val_dataset = create_torch_dataset("your_dataset_name")
```

## Training the Model

Training can be initiated through the command-line interface with various configuration options. The default settings are optimized for single-GPU training with 24GB of VRAM.

### Basic Training

For standard fine-tuning on your GUI dataset:

```bash
python train.py \
    --dataset_name your_fiftyone_dataset \
    --output_dir ./gui-actor-finetuned \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

### Memory-Optimized Training

If you encounter out-of-memory errors, use these optimized settings:

```bash
python train.py \
    --dataset_name your_fiftyone_dataset \
    --output_dir ./gui-actor-finetuned \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --model_max_length 8192 \
    --max_pixels 2048000 \
    --offload_to_cpu
```

### Single-Application Mode

For aggressive fine-tuning when targeting a specific application:

```bash
python train.py \
    --dataset_name your_fiftyone_dataset \
    --output_dir ./gui-actor-specialized \
    --single_app_mode \
    --unfreeze_vision_layers \
    --unfreeze_last_n_layers 8 \
    --num_train_epochs 5
```

## Understanding the Architecture

The implementation consists of several interconnected components that work together to enable efficient training:

### Dataset Transformation Pipeline

The transformation from FiftyOne annotations to GUI-Actor's training format happens in multiple stages. First, the `add_message_payload_to_dataset` function iterates through each annotation and constructs a three-message conversation containing a system prompt that instructs the model on its GUI agent role, a user message with the screenshot and task description, and an assistant response with the action and coordinates.

For keypoint annotations, the system extracts single-point coordinates and formats them with special pointer tokens. For detection annotations, it converts bounding boxes from FiftyOne's width-height format to the min-max format expected by the model, enabling region-based interactions.

### Coordinate Processing and Visual Token Alignment

One of the most critical aspects of the system is maintaining precise alignment between text coordinates and visual tokens. When the model processes an image, it divides it into patches that become visual tokens. The collator calculates which visual token contains each coordinate mentioned in the text, creating supervision signals that teach the model to ground its predictions in the visual input.

The `get_token_index` function performs this mapping by considering the image dimensions, patch size, and merge factor to determine the exact token position for any given coordinate. This ensures that when the model predicts a coordinate in text, it can be traced back to a specific region in the image.

### Memory Management Strategies

Training vision-language models requires careful memory management. The implementation employs several strategies to work within hardware constraints:

**Gradient Accumulation**: Instead of processing large batches that exceed memory limits, the system processes smaller micro-batches and accumulates gradients over multiple steps. This simulates larger batch sizes without the memory overhead.

**Selective Parameter Updates**: Rather than updating all model parameters, the system can selectively unfreeze only the components that need adaptation. This includes the pointer prediction head for coordinate generation, the language modeling head for response generation, and optionally the last few transformer layers for task-specific reasoning.

**Dynamic Memory Allocation**: By setting PyTorch's memory allocator to use expandable segments, the system reduces fragmentation and makes more efficient use of available GPU memory.

**Precision Optimization**: The model uses bfloat16 precision for computations, which reduces memory usage by half compared to float32 while maintaining better numerical stability than float16.

## Training Configuration Options

The training script provides extensive configuration options to customize the fine-tuning process:

### Model Architecture Parameters

- `--model_max_length`: Maximum sequence length for text processing (default: 8192)
- `--max_pixels`: Maximum pixel count for image processing (default: 2048000)
- `--pointer_loss_weight`: Weight for coordinate prediction loss (default: 1.0)
- `--lm_loss_weight`: Weight for language modeling loss (default: 1.0)

### Training Dynamics

- `--learning_rate`: Base learning rate for optimization (default: 2e-5)
- `--warmup_ratio`: Proportion of training for learning rate warmup (default: 0.1)
- `--weight_decay`: L2 regularization coefficient (default: 0.01)
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)

### Parameter Unfreezing Strategy

- `--unfreeze_all_parameters`: Enable training of all model parameters
- `--unfreeze_pointer_head`: Update pointer prediction head (default: True)
- `--unfreeze_lm_head`: Update language modeling head (default: True)
- `--unfreeze_last_n_layers`: Number of transformer layers to unfreeze (default: 4)
- `--unfreeze_vision_layers`: Update vision encoder layers for specialized tasks

## Monitoring Training Progress

The system provides comprehensive logging through TensorBoard:

```bash
tensorboard --logdir ./gui-actor-finetuned/logs
```

Monitor these key metrics during training:

- **Training Loss**: Should decrease steadily, with pointer loss and language modeling loss tracked separately
- **Gradient Norms**: Should remain stable; spikes indicate training instability
- **Learning Rate**: Follows the configured schedule with warmup and decay
- **GPU Memory Usage**: Helps identify opportunities for batch size optimization

## Using the Fine-Tuned Model

After training, the model can be loaded for inference:

```python
from transformers import AutoProcessor
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer

# Load the fine-tuned model
model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    "./gui-actor-finetuned",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("./gui-actor-finetuned")

# Process an image and instruction
from PIL import Image
image = Image.open("screenshot.png")
instruction = "Click on the search button"

inputs = processor(
    text=instruction,
    images=image,
    return_tensors="pt"
)

# Generate response with coordinates
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=False)
```

## Technical Considerations

When working with this implementation, keep in mind several important technical details:

**Coordinate Precision**: The system maintains coordinate precision through normalization to avoid exact 0 or 1 values, which can cause numerical instability. Coordinates are adjusted by a small epsilon value to ensure stable gradient flow.

**Visual Token Resolution**: The number of visual tokens depends on the image resolution and model configuration. Higher resolution images provide more precise localization but require more memory. The default settings balance precision with memory efficiency.

**Batch Size Trade-offs**: Smaller batch sizes reduce memory usage but can lead to noisier gradients. Gradient accumulation helps mitigate this by simulating larger batches across multiple forward passes.

**Multi-Scale Training**: For applications with varying screenshot resolutions, consider training with multiple image scales to improve generalization. The processor automatically handles resizing, but you may want to adjust `max_pixels` based on your data.

## Troubleshooting

If you encounter issues during training, these solutions address common problems:

**Out of Memory Errors**: Reduce `model_max_length` and `max_pixels`, increase `gradient_accumulation_steps`, or enable CPU offloading with `--offload_to_cpu`.

**Slow Training Speed**: Ensure Flash Attention 2 is properly installed, reduce the number of dataloader workers, and verify that gradient checkpointing is enabled.

**Poor Coordinate Accuracy**: Check that your annotations use normalized coordinates (0-1 range), ensure sufficient training data for each interaction type, and consider increasing the `pointer_loss_weight`.

**Model Not Learning**: Verify that appropriate parameters are unfrozen, check that the learning rate isn't too low, and ensure your dataset has consistent annotation quality.

## Contributing

Contributions are welcome! Areas where help would be particularly valuable include:

- Support for additional annotation types (polygons, masks)
- Multi-GPU training optimization
- Integration with other annotation platforms
- Performance benchmarking on different GUI domains

## License

This implementation builds upon Microsoft's GUI-Actor model and follows the same licensing terms. Please refer to the original GUI-Actor repository for complete license information.

## Acknowledgments

This work extends Microsoft's GUI-Actor research and leverages the FiftyOne platform for dataset management. Special thanks to the teams behind these foundational technologies that make GUI automation research more accessible.

# Citations

```bibtex
@article{wu2025gui,
  title={GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents},
  author={Wu, Qianhui and Cheng, Kanzhi and Yang, Rui and Zhang, Chaoyun and Yang, Jianwei and Jiang, Huiqiang and Mu, Jian and Peng, Baolin and Qiao, Bo and Tan, Reuben and others},
  journal={arXiv preprint arXiv:2506.03143},
  year={2025}
}
```

```
@article{xu2024aguvis,
  title={Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction},
  author={Yiheng Xu and Zekun Wang and Junli Wang and Dunjie Lu and Tianbao Xie and Amrita Saha and Doyen Sahoo and Tao Yu and Caiming Xiong},
  year={2024},
  url={https://arxiv.org/abs/2412.04454}
}
```
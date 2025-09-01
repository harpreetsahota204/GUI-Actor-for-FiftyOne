import os
import torch
import transformers
from transformers import AutoProcessor, TrainingArguments
import argparse

# Set PyTorch memory allocation environment variable to avoid fragmentation
# This was suggested in the error message
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.trainer import AGUVISTrainer  # Removed rank0_print and safe_save_model_for_hf_trainer
from gui_actor.dataset import GUIActorFiftyOneCollator
from gui_actor.constants import ADDITIONAL_SPECIAL_TOKENS
from gui_actor.create_fo_torch_dataset import create_torch_dataset

def optimize_memory_settings():
    """
    Configure PyTorch memory settings for optimal training efficiency.
    Call this function before initializing the model.
    """
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set PyTorch deterministic behavior for more consistent memory usage
    # (but may impact performance)
    torch.use_deterministic_algorithms(False)
    
    # Enable TF32 on Ampere+ GPUs (faster with minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Use BFloat16 where possible (better numerical stability than FP16)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # Set memory allocation strategy for CUDA
    if hasattr(torch.cuda, 'memory_stats'):
        print("Optimizing CUDA memory allocation strategy...")
        
    print("Memory optimization settings applied")

def train_gui_actor_on_fiftyone(
    train_dataset,
    val_dataset,
    output_dir="./gui-actor-fiftyone-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,  # Reduced to prevent OOM during eval
    learning_rate=2e-5,
    save_steps=500,
    logging_steps=50,
    gradient_accumulation_steps=16,  # Increased for better memory efficiency
    warmup_ratio=0.1,
    weight_decay=0.01,
    model_max_length=24576,  
    max_pixels=5720064,  
    pointer_loss_weight=1.0,
    lm_loss_weight=1.0,
    unfreeze_all_parameters=False,
    unfreeze_pointer_head=True,
    unfreeze_lm_head=True,
    unfreeze_last_n_layers=4,
    unfreeze_vision_layers=False,
    single_app_mode=False,
    max_grad_norm=1.0,  # Added to prevent gradient explosion
    offload_to_cpu=False  # Option to offload optimizer state to CPU
):
    # Apply memory optimization settings
    optimize_memory_settings()
    
    print("\n🚀 Starting GUI-Actor training with memory-optimized settings")
    print("💾 Memory optimization techniques applied:")
    print("  - Chunked tensor operations in forward pass")
    print("  - Gradient checkpointing enabled")
    print("  - PyTorch memory allocation optimized with expandable_segments")
    print("  - Reduced sequence and image dimensions")
    print("  - Increased gradient accumulation steps")
    print(f"  - Using batch size: {per_device_train_batch_size} with {gradient_accumulation_steps}x accumulation")
    
    # Load pretrained GUI-Actor
    model_name = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
    
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    diff_marker_id = tokenizer.convert_tokens_to_ids("<|diff_marker|>")
    model.config.eos_token_id = [diff_marker_id]
    tokenizer.eos_token = "<|diff_marker|>"
    tokenizer.eos_token_id = diff_marker_id
    
    # Also update generation config if it exists
    if hasattr(model, 'generation_config'):
        model.generation_config.eos_token_id = [diff_marker_id]

    print(f"Set EOS token to <|diff_marker|> (ID: {diff_marker_id}) for training and inference")

    # Set loss weights to match original recipe
    model.reset_loss_weights(pointer_loss_weight=pointer_loss_weight, lm_loss_weight=lm_loss_weight)
    model.config.use_cache = False
    
    # Set model max length and max pixels to match original
    model.config.max_position_embeddings = model_max_length
    if hasattr(model.config, 'max_pixels'):
        model.config.max_pixels = max_pixels
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    
    # Ensure special tokens are set
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply single-app mode overrides for aggressive fine-tuning
    if single_app_mode:
        print("🎯 SINGLE-APP MODE: Applying aggressive fine-tuning settings...")
        unfreeze_vision_layers = True
        unfreeze_last_n_layers = max(8, unfreeze_last_n_layers)  # At least 8 layers
        print(f"   - Vision layers will be unfrozen")
        print(f"   - Unfreezing {unfreeze_last_n_layers} transformer layers")
        print(f"   - This will maximize adaptation to your specific application")
    
    # Fine-tuning parameter unfreezing strategy
    if unfreeze_all_parameters:
        print("WARNING: Unfreezing all parameters - this may cause overfitting on small datasets!")
        for p in model.parameters():
            p.requires_grad = True
    else:
        # Selective unfreezing strategy for fine-tuning
        print("Using selective unfreezing strategy for fine-tuning...")
        
        # Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze task-specific heads
        if unfreeze_lm_head:
            print("Unfreezing language modeling head...")
            for p in model.lm_head.parameters():
                p.requires_grad = True
        
        if unfreeze_pointer_head and hasattr(model, 'pointer_head'):
            print("Unfreezing pointer head...")
            for p in model.pointer_head.parameters():
                p.requires_grad = True
        
        # Unfreeze last N transformer layers for adaptation
        if unfreeze_last_n_layers > 0:
            print(f"Unfreezing last {unfreeze_last_n_layers} transformer layers...")
            total_layers = len(model.language_model.layers)
            layers_to_unfreeze = model.language_model.layers[-unfreeze_last_n_layers:]
            for layer in layers_to_unfreeze:
                for p in layer.parameters():
                    p.requires_grad = True
        
        # Unfreeze vision layers for single-application fine-tuning
        if unfreeze_vision_layers:
            print("Unfreezing vision layers for single-application adaptation...")
            
            # Try different vision encoder architectures
            vision_unfrozen = False
            
            # For Qwen2.5-VL vision encoder with blocks structure
            if hasattr(model, 'visual') and hasattr(model.visual, 'blocks'):
                print("  - Unfreezing Qwen2.5-VL vision blocks...")
                # Unfreeze last 25% of vision blocks for single-app mode
                total_vision_blocks = len(model.visual.blocks)
                blocks_to_unfreeze = max(2, total_vision_blocks // 4)
                for block in model.visual.blocks[-blocks_to_unfreeze:]:
                    for p in block.parameters():
                        p.requires_grad = True
                print(f"    Unfroze last {blocks_to_unfreeze}/{total_vision_blocks} vision blocks")
                vision_unfrozen = True
            
            # Fallback: For other Qwen2.5-VL vision encoder structures
            elif hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
                print("  - Unfreezing Qwen2.5-VL vision transformer layers...")
                # Unfreeze last 25% of vision layers for single-app mode
                total_vision_layers = len(model.visual.transformer.resblocks)
                layers_to_unfreeze = max(2, total_vision_layers // 4)
                for layer in model.visual.transformer.resblocks[-layers_to_unfreeze:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                print(f"    Unfroze last {layers_to_unfreeze}/{total_vision_layers} vision layers")
                vision_unfrozen = True
            
            # Alternative: if vision encoder is structured differently
            elif hasattr(model, 'vision_model'):
                print("  - Unfreezing alternative vision model layers...")
                if hasattr(model.vision_model, 'encoder') and hasattr(model.vision_model.encoder, 'layers'):
                    total_layers = len(model.vision_model.encoder.layers)
                    layers_to_unfreeze = max(2, total_layers // 4)
                    for layer in model.vision_model.encoder.layers[-layers_to_unfreeze:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                    print(f"    Unfroze last {layers_to_unfreeze}/{total_layers} vision layers")
                    vision_unfrozen = True
            
            if not vision_unfrozen:
                print("  - Warning: Could not locate vision layers to unfreeze")
                print("  - Vision encoder architecture may be different than expected")
    
    # Enable gradient checkpointing with proper method
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled for memory efficiency")
    
    # Free up memory
    torch.cuda.empty_cache()
    
    # Print trainable parameters summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    # Memory usage info
    if torch.cuda.is_available():
        print(f"\nGPU Memory Stats:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"    - Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"    - Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")
    
    # Print which components are trainable
    print("\nTrainable components:")
    if any(p.requires_grad for p in model.lm_head.parameters()):
        print("- Language modeling head")
    if hasattr(model, 'pointer_head') and any(p.requires_grad for p in model.pointer_head.parameters()):
        print("- Pointer head")
    
    trainable_layers = []
    for i, layer in enumerate(model.language_model.layers):
        if any(p.requires_grad for p in layer.parameters()):
            trainable_layers.append(str(i))
    if trainable_layers:
        print(f"- Transformer layers: {', '.join(trainable_layers)}")
    
    # Create collator with both processor and tokenizer
    collator = GUIActorFiftyOneCollator(processor, tokenizer)
    
    # Training arguments optimized for memory efficiency
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=2,  # Reduced for disk space efficiency
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        
        # Memory optimization settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TF32 precision on Ampere+ GPUs
        fp16=False,  # Disable fp16 as bf16 is used
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=max_grad_norm,  # Prevent gradient explosion
        dataloader_num_workers=0,  # No additional workers to reduce memory
        dataloader_prefetch_factor=None,  # Disable prefetching
        
        # Additional memory optimizations
        optim="adamw_torch",  # Use PyTorch's AdamW implementation
        lr_scheduler_type="cosine",  # Cosine scheduler
        group_by_length=False,  # Disable grouping by length
        ddp_find_unused_parameters=False,  # Performance optimization
        
        # Memory monitoring
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,  # Log the first step to catch early OOMs
        
        # Offload optimizer state to CPU if requested
        optim_args=f"offload_optimizer={str(offload_to_cpu).lower()}" if offload_to_cpu else None,
    )
    
    # Initialize trainer
    trainer = AGUVISTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.create_optimizer = trainer.create_optimizer_with_different_learning_rates

    
    # Train
    trainer.train()
    
    # Save final model - use standard save_model method
    model.config.use_cache = True
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    return model, processor

def main():
    parser = argparse.ArgumentParser(description="Train GUI-Actor on FiftyOne dataset")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True,
        help="Name of the FiftyOne dataset to use for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gui-actor-fiftyone-finetuned",
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log metrics every N steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,  # Increased from 4 to 16 for memory efficiency
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio of total training steps"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient"
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=8192,  # Reduced from 24576 to 8192 for memory efficiency
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=2048000,  # Reduced from 5720064 to 2048000 for memory efficiency
        help="Maximum number of pixels for vision input"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offload optimizer states to CPU to save GPU memory"
    )
    parser.add_argument(
        "--pointer_loss_weight",
        type=float,
        default=1.0,
        help="Weight for pointer loss"
    )
    parser.add_argument(
        "--lm_loss_weight",
        type=float,
        default=1.0,
        help="Weight for language modeling loss"
    )
    parser.add_argument(
        "--unfreeze_all_parameters",
        action="store_true",
        help="Unfreeze all model parameters for training (not recommended for fine-tuning)"
    )
    parser.add_argument(
        "--unfreeze_pointer_head",
        action="store_true",
        default=True,
        help="Unfreeze pointer head for training"
    )
    parser.add_argument(
        "--unfreeze_lm_head",
        action="store_true", 
        default=True,
        help="Unfreeze language modeling head for training"
    )
    parser.add_argument(
        "--unfreeze_last_n_layers",
        type=int,
        default=4,
        help="Number of last transformer layers to unfreeze (0 to disable)"
    )
    parser.add_argument(
        "--unfreeze_vision_layers",
        action="store_true",
        help="Unfreeze some vision layers (recommended for single-application fine-tuning)"
    )
    parser.add_argument(
        "--single_app_mode",
        action="store_true",
        help="Enable aggressive fine-tuning for single application (unfreezes more components)"
    )
    
    args = parser.parse_args()
    
    # Create FiftyOne datasets
    print(f"Loading FiftyOne dataset: {args.dataset_name}")
    train_dataset, val_dataset = create_torch_dataset(args.dataset_name)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Print memory-efficiency notes
    print("\n📝 Using memory-efficient training settings:")
    print(f"  - Per-device batch size: {args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  - Max sequence length: {args.model_max_length}")
    print(f"  - Max pixels: {args.max_pixels}")
    print(f"  - GPU memory usage has been optimized to prevent OOM errors")
    print("  - See LOCAL_DEVELOPMENT.md for more details on memory optimizations")
    
    # Train the model with memory-optimized settings
    model, processor = train_gui_actor_on_fiftyone(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        model_max_length=args.model_max_length,
        max_pixels=args.max_pixels,
        pointer_loss_weight=args.pointer_loss_weight,
        lm_loss_weight=args.lm_loss_weight,
        unfreeze_all_parameters=args.unfreeze_all_parameters,
        unfreeze_pointer_head=args.unfreeze_pointer_head,
        unfreeze_lm_head=args.unfreeze_lm_head,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        unfreeze_vision_layers=args.unfreeze_vision_layers,
        single_app_mode=args.single_app_mode,
        max_grad_norm=args.max_grad_norm,
        offload_to_cpu=args.offload_to_cpu
    )
    
    print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
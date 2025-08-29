import torch
import transformers
from transformers import AutoProcessor, TrainingArguments
import argparse

from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.trainer import AGUVISTrainer  # Removed rank0_print and safe_save_model_for_hf_trainer
from gui_actor.dataset import GUIActorFiftyOneCollator
from gui_actor.constants import ADDITIONAL_SPECIAL_TOKENS
from gui_actor.create_fo_torch_dataset import create_torch_dataset

def train_gui_actor_on_fiftyone(
    train_dataset,
    val_dataset,
    output_dir="./gui-actor-fiftyone-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    save_steps=500,
    logging_steps=50,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_unfrozen_layers=6
):
    # Load pretrained GUI-Actor
    model_name = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
    
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    # Set loss weights for your use case
    model.reset_loss_weights(pointer_loss_weight=0.5, lm_loss_weight=1.0)
    model.config.use_cache = False
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    
    # Ensure special tokens are set
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    
    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze strategy: lm_head + last N transformer layers
    print("Unfreezing lm_head...")
    for p in model.lm_head.parameters():
        p.requires_grad = True
        
    print(f"Unfreezing last {num_unfrozen_layers} transformer layers...")
    for p in model.language_model.layers[-num_unfrozen_layers:].parameters():
        p.requires_grad = True
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    
    # Create collator with both processor and tokenizer
    collator = GUIActorFiftyOneCollator(processor, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        report_to="tensorboard",  # Add tensorboard logging
        logging_dir=f"{output_dir}/logs",
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
        default=2,
        help="Training batch size per device"
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
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--num_unfrozen_layers",
        type=int,
        default=6,
        help="Number of transformer layers to unfreeze from the end"
    )
    
    args = parser.parse_args()
    
    # Create FiftyOne datasets
    print(f"Loading FiftyOne dataset: {args.dataset_name}")
    train_dataset, val_dataset = create_torch_dataset(args.dataset_name)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Train the model
    model, processor = train_gui_actor_on_fiftyone(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_unfrozen_layers=args.num_unfrozen_layers
    )
    
    print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
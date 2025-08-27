import torch
import transformers
from transformers import AutoProcessor, TrainingArguments
from torch.utils.data import DataLoader

from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.trainer import AGUVISTrainer, rank0_print, safe_save_model_for_hf_trainer
from gui_actor.dataset import GUIActorFiftyOneCollator
from gui_actor.constants import ADDITIONAL_SPECIAL_TOKENS

def train_gui_actor_on_fiftyone(
    train_dataset,
    val_dataset,
    output_dir="./gui-actor-fiftyone-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    save_steps=500,
    logging_steps=50
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
    rank0_print("Unfreezing lm_head...")
    for p in model.lm_head.parameters():
        p.requires_grad = True
        
    rank0_print("Unfreezing last 6 transformer layers...")
    for p in model.model.layers[-6:].parameters():
        p.requires_grad = True
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    
    # Create collator
    collator = GUIActorFiftyOneCollator(processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
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
    
    # Save final model
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    rank0_print(f"Model saved to {output_dir}")
    
    return model, processor

# Usage
if __name__ == "__main__":
    # Your FiftyOne datasets
    from fiftyone.utils.torch import GetItem
    
    # Assuming you have train_view and val_view ready
    train_torch_dataset = train_view.to_torch(datagetter())
    val_torch_dataset = val_view.to_torch(datagetter())
    
    # Train
    model, processor = train_gui_actor_on_fiftyone(
        train_dataset=train_torch_dataset,
        val_dataset=val_torch_dataset,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-5
    )
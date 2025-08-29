"""
Simplified trainer class for single GPU training.

Stripped of:
- Distributed training components (rank0_print, dist)
- DeepSpeed/ZeRO-3 handling
- FSDP configuration
- Sagemaker support
- Complex accelerator setup
"""

from functools import wraps
from typing import Optional

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    get_parameter_names,
    has_length,
    is_datasets_available,
)
from transformers.trainer_pt_utils import LengthGroupedSampler as HFLengthGroupedSampler

# Only import if using FiftyOne datasets
try:
    from fiftyone.utils.torch import FiftyOneTorchDataset
    HAS_FIFTYONE = True
except ImportError:
    HAS_FIFTYONE = False

if is_datasets_available():
    import datasets


def seed_worker(worker_id):
    """Worker init function for DataLoader reproducibility."""
    import random
    import numpy as np
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AGUVISTrainer(Trainer):
    """
    Simplified trainer for single GPU training.
    
    Features:
    - Custom EOS token handling during saving
    - Different learning rates for different parameter groups
    - Optional FiftyOne dataset support
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store original save methods
        original_save = self._save
        original_save_model = self.save_model

        def modify_eos_token(func):
            """Decorator to temporarily modify EOS token during saving."""
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Store original token settings
                tokenizer = self.processing_class.tokenizer
                old_config_id = self.model.config.eos_token_id
                old_eos_token = tokenizer.eos_token
                old_generation_config_eos_token_id = (
                    self.model.generation_config.eos_token_id 
                    if hasattr(self.model, "generation_config") else None
                )

                try:
                    # Set custom EOS token
                    new_eos_token_id = tokenizer.convert_tokens_to_ids("<|diff_marker|>")
                    self.model.config.eos_token_id = [new_eos_token_id]
                    tokenizer.eos_token = "<|diff_marker|>"
                    if hasattr(self.model, "generation_config"):
                        self.model.generation_config.eos_token_id = [new_eos_token_id]

                    print("Set eos token id to", new_eos_token_id)
                    
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Restore original token settings
                    self.model.config.eos_token_id = old_config_id
                    tokenizer.eos_token = old_eos_token
                    if hasattr(self.model, "generation_config") and old_generation_config_eos_token_id is not None:
                        self.model.generation_config.eos_token_id = old_generation_config_eos_token_id

                    print("Set eos token id back to", old_config_id)

            return wrapper

        # Apply EOS token modification to save methods
        self._save = modify_eos_token(original_save)
        self.save_model = modify_eos_token(original_save_model)
    
    # Only override if using FiftyOne datasets
    if HAS_FIFTYONE:
        def get_train_dataloader(self):
            """Add FiftyOne's worker_init_fn for MongoDB handling."""
            dataloader = super().get_train_dataloader()
            
            if isinstance(dataloader, DataLoader):
                return DataLoader(
                    dataloader.dataset,
                    batch_size=dataloader.batch_size,
                    sampler=dataloader.sampler,
                    collate_fn=dataloader.collate_fn,
                    num_workers=dataloader.num_workers,
                    pin_memory=dataloader.pin_memory,
                    drop_last=dataloader.drop_last,
                    worker_init_fn=FiftyOneTorchDataset.worker_init,
                )
            return dataloader
        
        def get_eval_dataloader(self, eval_dataset=None):
            """Add FiftyOne's worker_init_fn for MongoDB handling."""
            dataloader = super().get_eval_dataloader(eval_dataset)
            
            if isinstance(dataloader, DataLoader):
                return DataLoader(
                    dataloader.dataset,
                    batch_size=dataloader.batch_size,
                    sampler=dataloader.sampler,
                    collate_fn=dataloader.collate_fn,
                    num_workers=dataloader.num_workers,
                    pin_memory=dataloader.pin_memory,
                    drop_last=dataloader.drop_last,
                    worker_init_fn=FiftyOneTorchDataset.worker_init,
                )
            return dataloader

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Get sampler for training data with optional length grouping."""
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return HFLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
            )
        elif getattr(self.args, 'group_by_modality_length', False):
            lengths = getattr(self.train_dataset, 'modality_lengths', self.train_dataset.lengths)
            return HFLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
            )
        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Remove unused columns based on dataset type
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # Configure dataloader parameters
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # Add sampler and worker settings for non-iterable datasets
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = (
                self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None
            )

        return DataLoader(train_dataset, **dataloader_params)

    def create_optimizer(self):
        """Create optimizer with weight decay applied selectively."""
        opt_model = self.model

        if self.optimizer is None:
            # Get parameters that should have weight decay
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Group parameters by weight decay application
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_optimizer_with_different_learning_rates(self):
        """
        Create optimizer with different learning rates for different parameter groups.
        
        Use this instead of create_optimizer() if you need different LRs for new parameters.
        """
        opt_model = self.model

        if self.optimizer is None:
            # Get parameters that should have weight decay
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # Identify new parameters that should use different learning rate
            new_parameters = []
            for name, param in opt_model.named_parameters():
                if ("pointer_head" in name) or ("embed_tokens" in name):
                    new_parameters.append(name)
            print(f"new_parameters: {len(new_parameters)}")
            
            # Group parameters
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() 
                              if ((n in decay_parameters) and (n not in new_parameters) and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() 
                              if ((n not in decay_parameters) and (n not in new_parameters) and p.requires_grad)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() 
                              if ((n in decay_parameters) and (n in new_parameters) and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                    "lr": getattr(self.args, 'learning_rate_new_params', self.args.learning_rate * 10),
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() 
                              if ((n not in decay_parameters) and (n in new_parameters) and p.requires_grad)],
                    "weight_decay": 0.0,
                    "lr": getattr(self.args, 'learning_rate_new_params', self.args.learning_rate * 10),
                },
            ]

            # Create optimizer without default learning rate
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.pop("lr")
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
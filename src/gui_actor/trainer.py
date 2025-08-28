"""
Custom trainer class extending Hugging Face's Trainer for GUI interaction tasks.

This module provides specialized training functionality for GUI interaction models,
including:
- Custom EOS token handling
- Accelerator setup
- Optimizers with different learning rates for different parameter groups
- Length-based batch grouping
"""

from datetime import timedelta
from functools import wraps
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer
from fiftyone.utils.torch import FiftyOneTorchDataset

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    get_parameter_names,
    has_length,
    is_accelerate_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_pt_utils import LengthGroupedSampler as HFLengthGroupedSampler
# Import removed - will define custom seed_worker function below
from transformers.utils import logging

if is_datasets_available():
    import datasets


def rank0_print(*args):
    """Print message only from rank 0 process in distributed training."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Handle DeepSpeed ZeRO-3 parameters.
    
    Gathers parameters across devices if using ZeRO-3 sharding.
    Returns CPU copy of the parameter.
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
    Get multimodal adapter parameters, handling ZeRO-3 if needed.
    
    Extracts parameters matching given keys and moves them to CPU.
    """
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Safely save model accounting for distributed training.
    
    Ensures all processes are synced before saving and handles DeepSpeed case.
    """
    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def seed_worker(worker_id):
    """
    Worker init function for DataLoader.
    
    Sets different random seed for each worker to ensure reproducibility
    while maintaining randomness across workers.
    
    Args:
        worker_id: ID of the worker process (passed by PyTorch DataLoader)
    """
    import random
    import numpy as np
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AGUVISTrainer(Trainer):
    """
    Custom trainer for GUI interaction tasks extending HF Trainer.
    
    Adds functionality for:
    - Custom EOS token handling during saving
    - Specialized accelerator setup
    - Different learning rates for different parameter groups
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize newer Trainer attributes for compatibility
        if not hasattr(self, 'is_tp_enabled'):
            self.is_tp_enabled = False
        if not hasattr(self, 'is_sagemaker_dp_enabled'):
            self.is_sagemaker_dp_enabled = False
        if not hasattr(self, 'is_sagemaker_mp_enabled'):
            self.is_sagemaker_mp_enabled = False

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
                    self.model.generation_config.eos_token_id if hasattr(self.model, "generation_config") else None
                )

                try:
                    # Set custom EOS token
                    new_eos_token_id = tokenizer.convert_tokens_to_ids("<|diff_marker|>")
                    self.model.config.eos_token_id = [new_eos_token_id]
                    tokenizer.eos_token = "<|diff_marker|>"
                    if hasattr(self.model, "generation_config"):
                        self.model.generation_config.eos_token_id = [new_eos_token_id]

                    print("Set eos token id to", new_eos_token_id)
                    print("Set eos token to", "<|diff_marker|>")
                    print("Set generation config eos token id to", [new_eos_token_id])

                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Restore original token settings
                    self.model.config.eos_token_id = old_config_id
                    tokenizer.eos_token = old_eos_token
                    if hasattr(self.model, "generation_config") and old_generation_config_eos_token_id is not None:
                        self.model.generation_config.eos_token_id = old_generation_config_eos_token_id

                    print("Set eos token id back to", old_config_id)
                    print("Set eos token back to", old_eos_token)
                    if old_generation_config_eos_token_id is not None:
                        print("Set generation config eos token id back to", old_generation_config_eos_token_id)

            return wrapper

        # Apply EOS token modification to save methods
        self._save = modify_eos_token(original_save)
        self.save_model = modify_eos_token(original_save_model)

    def get_train_dataloader(self):
        """
        Override to add FiftyOne worker_init_fn for proper MongoDB handling in multi-worker scenarios.
        """
        # Call parent's method to get the standard dataloader
        dataloader = super().get_train_dataloader()
        
        # If it's already a DataLoader, recreate it with worker_init_fn
        if isinstance(dataloader, DataLoader):
            return DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                collate_fn=dataloader.collate_fn,
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                worker_init_fn=FiftyOneTorchDataset.worker_init,  # Add FiftyOne worker init
            )
        return dataloader
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Override to add FiftyOne worker_init_fn for proper MongoDB handling in multi-worker scenarios.
        """
        # Call parent's method to get the standard dataloader
        dataloader = super().get_eval_dataloader(eval_dataset)
        
        # If it's already a DataLoader, recreate it with worker_init_fn
        if isinstance(dataloader, DataLoader):
            return DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=dataloader.sampler,
                collate_fn=dataloader.collate_fn,
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                worker_init_fn=FiftyOneTorchDataset.worker_init,  # Add FiftyOne worker init
            )
        return dataloader

    def create_accelerator_and_postprocess(self):
        """
        Create and configure accelerator for distributed training.
        
        Sets up gradient accumulation, process group timeouts, and FSDP/DeepSpeed if enabled.
        """
        # Configure gradient accumulation
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        # Set long timeout for process group
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))

        # Create accelerator with dataloader config
        dispatch_batches = getattr(self.args, "dispatch_batches", None)
        split_batches = getattr(self.args, "split_batches", None)
        self.dataloader_config = DataLoaderConfiguration(
            dispatch_batches=dispatch_batches,
            split_batches=split_batches,
        )
        self.accelerator = Accelerator(
            dataloader_config=self.dataloader_config,
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            kwargs_handlers=[accelerator_kwargs],
        )
        self.gather_function = self.accelerator.gather_for_metrics

        # Set distributed training flags
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # Configure FSDP if enabled
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        # Propagate args to DeepSpeed if needed
        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Get sampler for training data.
        
        Returns:
        - LengthGroupedSampler if grouping by length is enabled
        - RandomSampler otherwise
        - None if dataset has no length
        """
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
        """
        Create training DataLoader with appropriate configuration.
        
        Handles column removal, collation, and sampler setup.
        """
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

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Create optimizer with weight decay applied selectively.
        
        Separates parameters into groups with and without weight decay.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # Get parameters that should have weight decay
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Group parameters by weight decay application
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
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
        
        Groups parameters by:
        1. New vs existing parameters
        2. Weight decay vs no weight decay
        Each group gets appropriate learning rate and weight decay settings.
        """
        if is_sagemaker_mp_enabled():
            raise NotImplementedError("Sagemaker MP is not supported for separate learning rate yet")
            return super().create_optimizer()

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
            rank0_print(f"new_parameters: {len(new_parameters)}")
            
            # Group parameters by:
            # 1. Existing params with weight decay
            # 2. Existing params without weight decay  
            # 3. New params with weight decay
            # 4. New params without weight decay
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n in decay_parameters) and (n not in new_parameters) and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n not in decay_parameters) and (n not in new_parameters) and p.requires_grad)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n in decay_parameters) and (n in new_parameters) and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                    "lr": getattr(self.args, 'learning_rate_new_params', self.args.learning_rate * 10),
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n not in decay_parameters) and (n in new_parameters) and p.requires_grad)],
                    "weight_decay": 0.0,
                    "lr": getattr(self.args, 'learning_rate_new_params', self.args.learning_rate * 10),
                },
            ]

            # Create optimizer without default learning rate
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.pop("lr")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
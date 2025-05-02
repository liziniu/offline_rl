import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import (
    ###
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    is_torch_xla_available,
    SaveStrategy,
)
from typing import List, Optional, Dict


class OfflineREINFORCETrainer(Trainer):
    reward_clip_value = 5.0
    reduction: str = "num_items_in_batch"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        ref_log_probs = inputs.pop("ref_log_prob", None)
        rewards = inputs.pop("reward", None)
        if rewards is None:
            rewards = torch.zeros_like(inputs["input_ids"].size(0))
        else:
            rewards = torch.clamp(
                rewards, -self.reward_clip_value, self.reward_clip_value
            )

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()

        mask = (inputs["labels"] != -100)[:, 1:]
        log_probs = chunked_gather_log_probs(
            logits[:, :-1, :],
            inputs["input_ids"][:, 1:],
        )
        if ref_log_probs is not None:
            kl = (log_probs - ref_log_probs).detach()
        else:
            kl = torch.zeros_like(log_probs)

        advantages = torch.zeros_like(log_probs)
        for i in range(rewards.size(0)):
            advantages[i][:] = rewards[i]
            advantages[i] -= kl[i] * self.args.kl_coeff

        if self.reduction == "mean":
            losses = -(advantages * log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
            loss = losses.mean()
        elif self.reduction == "num_items_in_batch" and num_items_in_batch is not None:
            losses = -(advantages * log_probs * mask).sum(dim=-1)
            loss = losses.sum() / num_items_in_batch
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        if not self.control.should_evaluate:
            self.training_logs = {}

            kl_mean = torch.mean((kl * mask).sum(dim=-1) / mask.sum(dim=-1))
            self.training_logs["kl"] = round(kl_mean.item(), 4)

            with torch.no_grad():
                masked_logits = logits[:, :-1][mask]
                entropy = chunked_entropy_from_logits(
                    masked_logits, batch_size=max(1, len(masked_logits) // 4)
                )
                self.training_logs["entropy"] = round(torch.mean(entropy).item(), 4)

            self.training_logs["ce_loss"] = (
                outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            )
            self.training_logs["ce_loss"] = round(
                self.training_logs["ce_loss"].item(), 4
            )

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()
            logs.update(getattr(self, "training_logs", {}))

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


def chunked_gather_log_probs(logits, labels, chunk_size=None):
    if chunk_size is None:
        chunk_size = logits.size(0)
    # Total number of sequences (batch size)
    total_sequences = logits.size(0)
    # Initialize a tensor to hold the log probabilities results for each sequence and label
    log_probs_labels_results = torch.empty_like(
        labels, dtype=logits.dtype, device=logits.device
    )

    # Process in chunks to manage memory
    for start in range(0, total_sequences, chunk_size):
        end = start + chunk_size
        chunk_logits = logits[start:end]
        chunk_labels = labels[start:end]

        # Convert logits to log probabilities
        log_probs = F.log_softmax(chunk_logits, dim=-1)
        # Gather log probabilities at the label indices
        log_probs_labels = log_probs.gather(dim=-1, index=chunk_labels.unsqueeze(-1))

        # Store results
        log_probs_labels_results[start:end] = log_probs_labels.squeeze(-1)

    return log_probs_labels_results


def chunked_entropy_from_logits(chunk_logits, batch_size=None):
    """
    Compute entropy from logits in a memory-efficient manner by introducing a batch_size parameter.

    Args:
        chunk_logits (torch.Tensor): Logits tensor of shape (total_samples, num_classes).
        batch_size (int): Number of samples to process per batch.

    Returns:
        torch.Tensor: Entropy tensor of shape (total_samples,).
    """
    total_samples, num_classes = chunk_logits.shape
    entropy_list = []
    if batch_size is None:
        batch_size = total_samples

    # Process logits in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        logits_batch = chunk_logits[start_idx:end_idx]  # Get a batch of logits

        # Compute logsumexp for the current batch
        logsumexp_batch = torch.logsumexp(
            logits_batch, dim=-1, keepdim=False
        )  # Shape: (batch_size,)
        # Compute probabilities in log-space without computing softmax
        normalized_logits = logits_batch - logsumexp_batch.unsqueeze(
            -1
        )  # Shape: (batch_size, num_classes)
        exp_normalized_logits = torch.exp(
            normalized_logits
        )  # Shape: (batch_size, num_classes)
        # Compute entropy for the batch
        entropy_batch = logsumexp_batch - (logits_batch * exp_normalized_logits).sum(
            dim=-1
        )  # Shape: (batch_size,)

        entropy_list.append(entropy_batch)  # Store entropy for the current batch

    # Concatenate results from all batches
    if len(entropy_list) > 0:
        return torch.cat(entropy_list, dim=0)
    else:
        return torch.tensor(0.0)

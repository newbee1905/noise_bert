import torch
from transformers import Trainer


class JepaTrainer(Trainer):
	def training_step(self, model, inputs, num_items_in_batch=None):
		# Let the standard HF Trainer handle the forward pass, backward pass,
		# gradient accumulation, and mixed precision scaling safely.
		# This prevents AssertionError regarding GradScaler in fp16 mode.
		loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

		# Safely extract the original model if it was wrapped by Accelerate/DDP
		unwrapped_model = self.accelerator.unwrap_model(model)

		# Apply the EMA update
		momentum = getattr(self.args, "ema_momentum", 0.99)
		unwrapped_model.update_ema(momentum=momentum)

		return loss

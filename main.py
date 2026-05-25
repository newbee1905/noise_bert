import os
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoConfig
from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapRandomCharacterSubstitution, CompositeTransformation
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from custom_attacks import LeetSpeakWordSwap, PunctuationInsertionSwap

from models import DenoisingRobustModel
from trainer.jepa import JepaTrainer
from trainer.dataset import DynamicRobustDataset


def compute_metrics(eval_pred):
	logits, labels = eval_pred
	if isinstance(logits, tuple):
		logits = logits[0]
	predictions = np.argmax(logits, axis=-1)
	return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}


def prepare_dataset(cfg: DictConfig, tokenizer):
	print(f"\nLoading Dataset for {cfg.tasks.name.upper()}...")
	raw_datasets = load_dataset(cfg.tasks.dataset_name, cfg.tasks.dataset_config)

	train_dataset = DynamicRobustDataset(raw_datasets["train"], tokenizer, cfg)

	# The validation dataset must remain 100% clean (unaugmented) to correctly
	# measure baseline semantic accuracy and the "Robustness Tax".
	def val_map(examples):
		if len(cfg.tasks.text_columns) == 2:
			enc = tokenizer(
				examples[cfg.tasks.text_columns[0]],
				examples[cfg.tasks.text_columns[1]],
				padding="max_length",
				truncation=True,
				max_length=cfg.tasks.max_length,
			)
		else:
			enc = tokenizer(
				examples[cfg.tasks.text_columns[0]],
				padding="max_length",
				truncation=True,
				max_length=cfg.tasks.max_length,
			)
		res = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": examples["label"]}
		B = len(examples["label"])
		res["clean_input_ids"] = enc["input_ids"]
		res["clean_attention_mask"] = enc["attention_mask"]
		res["noisy_typo_mask"] = torch.zeros(B, cfg.tasks.max_typos, cfg.tasks.max_length).tolist()
		res["clean_typo_mask"] = torch.zeros(B, cfg.tasks.max_typos, cfg.tasks.max_length).tolist()
		return res

	eval_dataset = raw_datasets["validation"].map(
		val_map, batched=True, remove_columns=raw_datasets["validation"].column_names
	)
	return train_dataset, eval_dataset


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))
	wandb.init(
		project="noise-bert",
		name=f"{cfg.tasks.name}-{cfg.models.ssl_method}-{cfg.models.classification_mode}-{cfg.models.base_model}",
	)
	tokenizer = AutoTokenizer.from_pretrained(cfg.models.base_model)
	train_dataset, eval_dataset = prepare_dataset(cfg, tokenizer)
	config = AutoConfig.from_pretrained(cfg.models.base_model, num_labels=cfg.tasks.num_labels)
	model = DenoisingRobustModel(
		config,
		model_name=cfg.models.base_model,
		ssl_method=cfg.models.ssl_method,
		projector_dim=cfg.models.projector_dim,
		classification_mode=cfg.models.classification_mode,
	)
	training_args = TrainingArguments(
		output_dir=cfg.output_dir,
		num_train_epochs=cfg.tasks.epochs,
		per_device_train_batch_size=cfg.tasks.batch_size,
		per_device_eval_batch_size=cfg.tasks.batch_size * 2,
		learning_rate=cfg.tasks.learning_rate,
		eval_strategy="epoch",
		save_strategy="epoch",
		logging_steps=cfg.tasks.logging_steps,
		report_to="wandb",
		load_best_model_at_end=True,
		metric_for_best_model="accuracy",
		save_total_limit=1,  # Keep only the best checkpoint to save space
		fp16=torch.cuda.is_available(),
		seed=cfg.seed,
		dataloader_num_workers=8,
	)
	training_args.ema_momentum = cfg.models.ema_momentum
	trainer = JepaTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
	)
	trainer.train()
	trainer.save_model(os.path.join(cfg.output_dir, "final"))


if __name__ == "__main__":
	main()

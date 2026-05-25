import torch
import textattack
import os
import threading
import logging
import re
import json
from concurrent.futures import ThreadPoolExecutor
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BertConfig,
	BertForMaskedLM,
	BertTokenizer,
	AutoConfig,
)
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import Pruthi2019, DeepWordBugGao2018, TextFoolerJin2019
from spellchecker import SpellChecker
from safetensors.torch import load_file

from models import DenoisingRobustModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

gpu_lock = threading.Lock()


class DenoisingDefense:
	def __init__(self, predictor_model, tokenizer, device="cuda", max_typos=15, max_length=128):
		self.model = predictor_model
		self.tokenizer = tokenizer
		self.device = device
		self.spell = SpellChecker()
		self.max_typos = max_typos
		self.max_length = max_length

	def get_typo_indices(self, text):
		"""Uses spellcheck to find potential typo words and their token indices."""
		# Handle pair tasks (like QNLI) where text might be a tuple
		if isinstance(text, (list, tuple)):
			text_list = text
		else:
			text_list = [text]

		# Tokenize early to get sequence IDs and word IDs
		if len(text_list) == 2:
			enc = self.tokenizer(text_list[0], text_list[1], truncation=True, max_length=self.max_length)
		else:
			enc = self.tokenizer(text_list[0], truncation=True, max_length=self.max_length)

		word_ids = enc.word_ids()
		seq_ids = enc.sequence_ids()

		# [max_typos, L]
		noisy_typo_mask = torch.zeros(self.max_typos, self.max_length, device=self.device)
		typo_counter = 0

		for seq_idx, single_text in enumerate(text_list):
			words = single_text.split()
			for i, word in enumerate(words):
				if typo_counter >= self.max_typos:
					break

				# Clean punctuation for checking
				clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())
				if clean_word and clean_word not in self.spell:
					# Map this specific word in this specific sequence to tokens
					for t_idx, (wid, sid) in enumerate(zip(word_ids, seq_ids)):
						if wid == i and sid == seq_idx and t_idx < self.max_length:
							noisy_typo_mask[typo_counter, t_idx] = 1.0
					typo_counter += 1

		return noisy_typo_mask.unsqueeze(0)  # [1, T, L]

	def defend(self, text):
		"""
		The key idea: The model's CLS token is vulnerable to noise.
		We use the predictor to 'denoise' the CLS token by cross-attending to the typo tokens.
		In this implementation, the forward pass of DenoisingRobustModel ALREADY does this
		if we provide the noisy_typo_mask.
		"""
		# However, for TextAttack compatibility, we need to return a text string or
		# modify the model wrapper to pass the mask.
		# Since we can't easily 'clean' the text string itself using this method
		# (it cleans the LATENT space), we must handle this in the ModelWrapper.
		return text


class DenoisingModelWrapper(HuggingFaceModelWrapper):
	def __init__(self, model, tokenizer, defender, use_predictor=True):
		super().__init__(model, tokenizer)
		self.defender = defender
		self.use_predictor = use_predictor

	def __call__(self, text_input_list):
		inputs_dict = self.tokenizer(
			text_input_list,
			add_special_tokens=True,
			padding="max_length",
			truncation=True,
			max_length=self.defender.max_length,
			return_tensors="pt",
		)
		inputs_dict = {k: v.to(self.defender.device) for k, v in inputs_dict.items()}

		if self.use_predictor:
			# For each text in batch, generate the typo mask via spellcheck
			masks = []
			for text in text_input_list:
				mask = self.defender.get_typo_indices(text)
				masks.append(mask)
			inputs_dict["noisy_typo_mask"] = torch.cat(masks, dim=0)
		else:
			# Explicitly pass None to bypass predictor logic in model.forward
			inputs_dict["noisy_typo_mask"] = None

		with torch.no_grad():
			outputs = self.model(**inputs_dict)

		return outputs.logits.cpu().numpy()


def load_any_model(model_path, task_name="sst2"):
	with gpu_lock:
		is_custom = os.path.exists(os.path.join(model_path, "model.safetensors"))

		if is_custom:
			config_path = os.path.join(model_path, "config.json")
			with open(config_path, "r") as f:
				config_dict = json.load(f)

			if "DenoisingRobustModel" in config_dict.get("architectures", []):
				logger.info(f"Loading custom DenoisingRobustModel from {model_path}...")
				config = AutoConfig.from_pretrained(model_path)
				model_type = config_dict.get("model_type", "bert")
				model_name_base = "roberta-base" if model_type == "roberta" else "bert-base-uncased"

				# We need to know the ssl_method to init correctly
				# For now, assume JEPA or detect from path
				if "sigreg" in model_path.lower():
					ssl_method = "sigreg"
				elif "fwp" in model_path.lower():
					ssl_method = "fwp"
				else:
					ssl_method = "jepa"

				model = DenoisingRobustModel(config, model_name=model_name_base, ssl_method=ssl_method)
				state_dict = load_file(os.path.join(model_path, "model.safetensors"))
				model.load_state_dict(state_dict, strict=False)
				tokenizer = AutoTokenizer.from_pretrained(model_name_base)

				return model, tokenizer

		# Standard fallback
		model = AutoModelForSequenceClassification.from_pretrained(model_path)
		tokenizer = AutoTokenizer.from_pretrained(model_path)
		return model, tokenizer


def run_single_attack(model_path, task_name, attack_recipe, num_examples=None, use_predictor=True):
	try:
		model, tokenizer = load_any_model(model_path, task_name)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		model.eval()

		if "DenoisingRobustModel" in str(type(model)):
			defender = DenoisingDefense(model, tokenizer, device=device)
			model_wrapper = DenoisingModelWrapper(model, tokenizer, defender, use_predictor=use_predictor)
		else:
			model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

		split = "validation"
		dataset = HuggingFaceDataset("glue", task_name, split=split)

		if attack_recipe == "pruthi":
			attack = Pruthi2019.build(model_wrapper)
		elif attack_recipe == "gao":
			attack = DeepWordBugGao2018.build(model_wrapper)
		elif attack_recipe == "textfooler":
			attack = TextFoolerJin2019.build(model_wrapper)
		else:
			from custom_attacks import build_custom_attack

			attack = build_custom_attack(model_wrapper)

		log_name = f"eval_{model_path.replace('/', '_').replace('.', '')}_{task_name}_{attack_recipe}"
		if not use_predictor:
			log_name += "_nopredictor"
		attack_args = AttackArgs(
			num_examples=num_examples if num_examples is not None else -1,
			log_to_csv=f"{log_name}_log.csv",
			checkpoint_interval=None,
			disable_stdout=True,
		)

		attacker = Attacker(attack, dataset, attack_args)
		results = attacker.attack_dataset()

		# Simple summary calculation
		num_success = sum(1 for r in results if "Successful" in type(r).__name__)
		num_failed = sum(1 for r in results if "Failed" in type(r).__name__)
		num_skipped = sum(1 for r in results if "Skipped" in type(r).__name__)
		total = len(results)

		clean_acc = ((total - num_skipped) / total) * 100 if total > 0 else 0
		adv_acc = (num_failed / total) * 100 if total > 0 else 0

		logger.info(f"MODEL: {model_path} | ATTACK: {attack_recipe} | Clean: {clean_acc:.2f}% | Adv: {adv_acc:.2f}%")

		return {"clean_acc": clean_acc, "adv_acc": adv_acc}
	except Exception as e:
		logger.error(f"Error: {e}")
		import traceback

		logger.error(traceback.format_exc())
		return None

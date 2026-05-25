import torch
import random
import numpy as np
from torch.utils.data import Dataset
from textattack.transformations import WordSwapRandomCharacterSubstitution
from custom_attacks import LeetSpeakWordSwap, PunctuationInsertionSwap


class DynamicRobustDataset(Dataset):
	def __init__(self, raw_dataset, tokenizer, cfg):
		self.raw_dataset = raw_dataset
		self.tokenizer = tokenizer
		self.cfg = cfg

		aug_type = getattr(cfg.tasks, "augmentation_type", "all")

		if aug_type == "typo" or aug_type == "char_swap":
			self.transformations = [WordSwapRandomCharacterSubstitution()]
		elif aug_type == "leetspeak":
			self.transformations = [LeetSpeakWordSwap()]
		elif aug_type == "punctuation":
			self.transformations = [PunctuationInsertionSwap()]
		else:
			self.transformations = [
				WordSwapRandomCharacterSubstitution(),
				LeetSpeakWordSwap(),
				PunctuationInsertionSwap(),
			]

	def __len__(self):
		return len(self.raw_dataset)

	def _augment_text(self, text):
		words = text.split()
		if not words:
			return text, []

		# Calculate number of typos: 30% capped at max_typos
		n = max(1, int(self.cfg.tasks.pct_words_to_swap * len(words)))
		n = min(n, self.cfg.tasks.max_typos)

		# Filter for words that are likely perturbable (avoiding just symbols)
		perturbable_indices = [i for i, w in enumerate(words) if any(c.isalnum() for c in w)]
		if not perturbable_indices:
			return text, []

		indices = random.sample(perturbable_indices, min(n, len(perturbable_indices)))

		aug_words = words.copy()
		typo_info = []

		for idx in indices:
			word = words[idx]
			random.shuffle(self.transformations)
			success = False
			for trans in self.transformations:
				replacements = trans._get_replacement_words(word)
				if replacements:
					new_word = random.choice(replacements)
					if new_word != word:
						aug_words[idx] = new_word
						typo_info.append(idx)
						success = True
						break

		return " ".join(aug_words), typo_info

	def __getitem__(self, idx):
		example = self.raw_dataset[idx]
		text_cols = self.cfg.tasks.text_columns

		clean_texts = [example[col] for col in text_cols]
		aug_texts = []
		all_typo_indices = []

		for t in clean_texts:
			aug_t, typo_idxs = self._augment_text(t)
			aug_texts.append(aug_t)
			all_typo_indices.append(typo_idxs)

		# Tokenize Noisy and Clean views
		if len(text_cols) == 2:
			aug_enc = self.tokenizer(
				aug_texts[0], aug_texts[1], padding="max_length", truncation=True, max_length=self.cfg.tasks.max_length
			)
			clean_enc = self.tokenizer(
				clean_texts[0],
				clean_texts[1],
				padding="max_length",
				truncation=True,
				max_length=self.cfg.tasks.max_length,
			)
		else:
			aug_enc = self.tokenizer(
				aug_texts[0], padding="max_length", truncation=True, max_length=self.cfg.tasks.max_length
			)
			clean_enc = self.tokenizer(
				clean_texts[0], padding="max_length", truncation=True, max_length=self.cfg.tasks.max_length
			)

		# Generate typo masks
		MAX_BUFFER = self.cfg.tasks.max_typos
		noisy_masks = torch.zeros(MAX_BUFFER, self.cfg.tasks.max_length)
		clean_masks = torch.zeros(MAX_BUFFER, self.cfg.tasks.max_length)

		typo_counter = 0
		for col_idx in range(len(text_cols)):
			a_word_ids = aug_enc.word_ids(batch_index=0)
			c_word_ids = clean_enc.word_ids(batch_index=0)
			a_seq_ids = aug_enc.sequence_ids(batch_index=0)
			c_seq_ids = clean_enc.sequence_ids(batch_index=0)

			for w_idx in all_typo_indices[col_idx]:
				if typo_counter >= MAX_BUFFER:
					break

				# Find subword tokens for this word in BOTH sequences
				# Match both word_index AND sequence_index (0 for first sent, 1 for second)
				a_tokens = [
					t_idx
					for t_idx, (wid, sid) in enumerate(zip(a_word_ids, a_seq_ids))
					if wid == w_idx and sid == col_idx and t_idx < self.cfg.tasks.max_length
				]
				c_tokens = [
					t_idx
					for t_idx, (wid, sid) in enumerate(zip(c_word_ids, c_seq_ids))
					if wid == w_idx and sid == col_idx and t_idx < self.cfg.tasks.max_length
				]

				if a_tokens:
					for t in a_tokens:
						noisy_masks[typo_counter, t] = 1
					for t in c_tokens:
						clean_masks[typo_counter, t] = 1
					typo_counter += 1

		return {
			"input_ids": torch.tensor(aug_enc["input_ids"]),
			"attention_mask": torch.tensor(aug_enc["attention_mask"]),
			"clean_input_ids": torch.tensor(clean_enc["input_ids"]),
			"clean_attention_mask": torch.tensor(clean_enc["attention_mask"]),
			"noisy_typo_mask": noisy_masks,
			"clean_typo_mask": clean_masks,
			"labels": torch.tensor(example["label"]),
		}

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from loss import SIGReg
from utils import Projector


class TransformerDecoderPredictor(nn.Module):
	"""
	Predictor: 3-layer Transformer Decoder with Cross-Attention.
	Inputs: CLS token + MASK tokens (Query) | Typo embeddings (Memory).
	"""

	def __init__(self, dim=768, nhead=12, num_layers=3):
		super().__init__()
		decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4, batch_first=True)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
		self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
		self.pad_token = nn.Parameter(torch.randn(1, 1, dim))

	def forward(self, cls_token, typo_embs, num_typos):
		B, D = cls_token.shape
		max_typos = typo_embs.shape[1]

		# Construct Query sequence: [B, max_typos + 1, D]
		queries = []
		mask_token_sq = self.mask_token.squeeze(0)  # [1, D]
		pad_token_sq = self.pad_token.squeeze(0)  # [1, D]

		for i in range(B):
			n = int(num_typos[i].item())
			q_i = [cls_token[i : i + 1]]
			if n > 0:
				q_i.append(mask_token_sq.expand(n, -1))
			if max_typos > n:
				q_i.append(pad_token_sq.expand(max_typos - n, -1))
			queries.append(torch.cat(q_i, dim=0))

		query = torch.stack(queries, dim=0)  # [B, max_typos + 1, D]

		memory_key_padding_mask = torch.zeros(B, max_typos, dtype=torch.bool, device=cls_token.device)
		tgt_key_padding_mask = torch.zeros(B, max_typos + 1, dtype=torch.bool, device=cls_token.device)
		for i in range(B):
			n = int(num_typos[i].item())
			if n < max_typos:
				memory_key_padding_mask[i, n:] = True
				tgt_key_padding_mask[i, n + 1 :] = True

		out = self.decoder(
			query, typo_embs, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
		)
		return out


class DenoisingRobustModel(PreTrainedModel):
	def __init__(
		self,
		config,
		model_name="bert-base-uncased",
		ssl_method="jepa",
		projector_dim="768-1536-128",
		classification_mode="clean_only",
	):
		super().__init__(config)
		self.encoder = AutoModel.from_pretrained(model_name)
		self.ssl_method = ssl_method
		self.classification_mode = classification_mode
		dim = config.hidden_size

		self.classifier = nn.Linear(dim, config.num_labels)

		# Predictor works in latent space BEFORE projection
		if ssl_method in ["jepa", "sigreg"]:
			self.predictor = TransformerDecoderPredictor(dim=dim)
			self.projector = Projector(projector_dim)

		if ssl_method == "jepa":
			self.target_encoder = AutoModel.from_pretrained(model_name)
			for param in self.target_encoder.parameters():
				param.requires_grad = False
		elif ssl_method == "sigreg":
			self.sigreg_loss = SIGReg()

		self.post_init()

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if hasattr(module, "bias") and module.bias is not None:
				module.bias.data.zero_()

	def update_ema(self, momentum=0.99):
		if self.ssl_method == "jepa":
			with torch.no_grad():
				for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
					param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		clean_input_ids=None,
		clean_attention_mask=None,
		noisy_typo_mask=None,
		clean_typo_mask=None,
		labels=None,
		**kwargs,
	):
		# Inference / Validation Path
		if clean_input_ids is None:
			outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
			latent = outputs.last_hidden_state
			cls_token = latent[:, 0, :]

			# FWP mode bypasses the predictor entirely
			if self.ssl_method == "fwp":
				logits = self.classifier(cls_token)
			else:
				has_typos = (noisy_typo_mask.sum() > 0) if noisy_typo_mask is not None else False
				if not has_typos:
					logits = self.classifier(cls_token)
				else:
					typo_counts = (noisy_typo_mask.sum(dim=-1) > 0).sum(dim=-1)
					typo_embs = torch.matmul(noisy_typo_mask.to(latent.dtype), latent)
					typo_token_counts = noisy_typo_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
					typo_embs = typo_embs / typo_token_counts

					preds = self.predictor(cls_token, typo_embs, typo_counts)
					cls_predicted = preds[:, 0, :]
					logits = self.classifier(cls_predicted)

			loss = None
			if labels is not None:
				loss = F.cross_entropy(logits, labels)

			return SequenceClassifierOutput(loss=loss, logits=logits)

		# Training Path
		outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
		latent_noisy = outputs.last_hidden_state
		cls_noisy = latent_noisy[:, 0, :]

		loss = torch.tensor(0.0, device=input_ids.device)

		# FWP Mode: Standard Adversarial Training (Train classifier on noisy input)
		if self.ssl_method == "fwp":
			logits = self.classifier(cls_noisy)
			if labels is not None:
				loss += F.cross_entropy(logits, labels)
			return SequenceClassifierOutput(loss=loss, logits=logits)

		# Target Clean Forward (For JEPA/SIGReg)

		if self.ssl_method == "jepa":
			with torch.no_grad():
				t_outputs = self.target_encoder(input_ids=clean_input_ids, attention_mask=clean_attention_mask)
				t_latent = t_outputs.last_hidden_state
		else:
			t_outputs = self.encoder(input_ids=clean_input_ids, attention_mask=clean_attention_mask)
			t_latent = t_outputs.last_hidden_state

		cls_clean = t_latent[:, 0, :]

		# Supervised Loss
		logits_clean = self.classifier(cls_clean)
		loss = torch.tensor(0.0, device=input_ids.device)

		# Determine training logits based on mode
		if self.classification_mode == "both":
			# Predictor must be run before loss calculation to get cls_predicted
			typo_counts = (noisy_typo_mask.sum(dim=-1) > 0).sum(dim=-1)
			typo_embs = torch.matmul(noisy_typo_mask.to(latent_noisy.dtype), latent_noisy)
			typo_token_counts = noisy_typo_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
			typo_embs = typo_embs / typo_token_counts

			preds = self.predictor(cls_noisy, typo_embs, typo_counts)
			cls_predicted = preds[:, 0, :]
			logits_pred = self.classifier(cls_predicted)

			if labels is not None:
				loss += (F.cross_entropy(logits_clean, labels) + F.cross_entropy(logits_pred, labels)) / 2.0

			# Still return clean logits for HF internal metrics
			logits = logits_clean
		else:
			# ONLY on clean CLS
			if labels is not None:
				loss += F.cross_entropy(logits_clean, labels)
			logits = logits_clean

		# SSL Objective (Predict noisy -> clean)
		if self.classification_mode != "both":
			typo_counts = (noisy_typo_mask.sum(dim=-1) > 0).sum(dim=-1)
			typo_embs = torch.matmul(noisy_typo_mask.to(latent_noisy.dtype), latent_noisy)
			typo_token_counts = noisy_typo_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
			typo_embs = typo_embs / typo_token_counts

			preds = self.predictor(cls_noisy, typo_embs, typo_counts)

		clean_typo_embs = torch.matmul(clean_typo_mask.to(t_latent.dtype), t_latent)
		clean_typo_token_counts = clean_typo_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
		clean_typo_embs = clean_typo_embs / clean_typo_token_counts

		targets = torch.cat([t_latent[:, 0:1, :], clean_typo_embs], dim=1)

		B, T, D = preds.shape
		valid_mask = torch.zeros(B, T, device=input_ids.device)
		for i in range(B):
			valid_mask[i, : typo_counts[i] + 1] = 1.0

		if self.ssl_method == "jepa":
			mse_elements = F.mse_loss(preds, targets, reduction="none")
			loss += (mse_elements.mean(dim=-1) * valid_mask).sum() / valid_mask.sum()

		elif self.ssl_method == "sigreg":
			mse_elements = F.mse_loss(preds, targets, reduction="none")
			loss += (mse_elements.mean(dim=-1) * valid_mask).sum() / valid_mask.sum()

			z_clean_cls = self.projector(targets[:, 0, :]).unsqueeze(1)
			loss += 10 * self.sigreg_loss(z_clean_cls)

		return SequenceClassifierOutput(loss=loss, logits=logits)

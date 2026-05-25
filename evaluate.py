import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from eval_utils import run_single_attack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))

	task_name = cfg.tasks.dataset_config
	attacks = cfg.tasks.get("attacks", [])
	model_paths = cfg.tasks.get("model_paths", [])
	num_examples = cfg.tasks.get("num_examples", 100)
	use_predictor = cfg.tasks.get("use_predictor", True)

	if not attacks or not model_paths:
		logger.error("No attacks or model_paths defined in the task configuration.")
		return

	for model_path in model_paths:
		for attack in attacks:
			logger.info(f"\n{'=' * 50}\nEvaluating: {model_path}\nAttack: {attack}\n{'=' * 50}")
			run_single_attack(
				model_path=model_path,
				task_name=task_name,
				attack_recipe=attack,
				num_examples=num_examples,
				use_predictor=use_predictor,
			)


if __name__ == "__main__":
	main()

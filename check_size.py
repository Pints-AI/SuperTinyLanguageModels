"""
Check model parameter count
"""
import hydra
from models.build_models import build_model
from models.utils import print_model_stats

@hydra.main(config_path="configs/training_configs", config_name="baseline")
def main(cfg):
    if "full_configs" in cfg:
        cfg = cfg["full_configs"]
    model = build_model(model_cfg=cfg["model"])[0]

    # print full parameter count
    print_model_stats(model)



if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
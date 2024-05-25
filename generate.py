"""
The main generate code
"""

import hydra
import torch

from models.build_models import build_model
from models.generator import StandardGenerator


@hydra.main(config_path="configs/generator", config_name="baseline")
def main(cfg):
    """run the main eval loop"""

    # set the checkpoint path to absolute path
    cfg["ckpt_path"] = hydra.utils.to_absolute_path(cfg["ckpt_path"])

    # load checkpoint from the path
    model = build_model(checkpoint=torch.load(cfg["ckpt_path"]))

    generator = StandardGenerator(model=model, generate_cfg=cfg)

    # generate the text
    generated_text = generator.default_generate(input_text=cfg["input_text"])

    print(generated_text)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter

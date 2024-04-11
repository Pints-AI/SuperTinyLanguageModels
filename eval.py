"""
The main eval code
"""
import hydra
import torch 

from evals.load_evaluators import load_evaluator





from trainers.utils import create_folder_structures
from models.build_models import build_model
from evals.load_evaluators import load_evaluator


@hydra.main(config_path="configs/", config_name="eval")
def main(cfg):
    """ run the main eval loop """

    # set the checkpoint path to absolute path
    cfg["checkpoint_path"] = hydra.utils.to_absolute_path(
        cfg["checkpoint_path"]
    )

    # load checkpoint from the path
    model = build_model(
        model_checkpoint=torch.load(cfg["checkpoint_path"])
    )

    # load the evaluator
    evaluator = load_evaluator(
        evaluator_name=cfg["evaluator_name"],
        cfg=cfg,
        model=model
    )

    # run the evaluator
    evaluator.evaluate(
        benchmark_names=cfg["benchmarks"]
    )



if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
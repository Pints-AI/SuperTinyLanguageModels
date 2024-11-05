"""
The main eval code
"""

import hydra
import torch
import evals 
from tqdm import tqdm
from models.build_models import build_model


@hydra.main(config_path="configs/test", config_name="test_stlm")
def main(cfg):
    """run the main eval loop"""

    # load checkpoint from the path if there
    if "model_ckpt" in cfg:
        # set the checkpoint path to absolute path
        cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])

        model, _ = build_model(checkpoint_path=cfg["model_ckpt"])
    # otherwise build the model from scratch (e.g. for external pretrained models)
    else:
        model, _ = build_model(model_cfg=cfg["model"])

    model.to(torch.device("cuda"))
    model.eval()

    benchmarks = cfg["benchmarks"]
    results_list = []

    # Outer progress bar for benchmarks
    with tqdm(benchmarks, desc="Evaluating benchmarks", position=0,  leave=True) as benchmark_bar:
        for benchmark in benchmark_bar:
            # Create the benchmark evaluator
            benchmark_evaluator = evals.make(benchmark)

            # Evaluating within the benchmark, tqdm already exists in the yield function
            results = benchmark_evaluator.evaluate(model=model)
            results_list.append(results)
            print(results)

    input(results_list)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter

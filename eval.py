from contextlib import nullcontext

import hydra
import torch
from hydra.utils import get_original_cwd
from models.build_models import build_model
from omegaconf import DictConfig

from evals import (
    arc,
    hellaswag,
    mmlu,
    model_wrapper,
    mteb_benchmark,
    nonsense,
    vitaminc,
    winograd,
)


def get_benchmark_class(name):
    """Loads in the class for a given benchmark."""
    if name == "vitaminc":
        return vitaminc.VitaminC
    elif name == "mteb":
        return mteb_benchmark.MTEBBenchmark
    elif name == "arc":
        return arc.ARC
    elif name == "winograd":
        return winograd.Winograd
    elif name == "nonsense":
        return nonsense.Nonsense
    elif name == "mmlu":
        return mmlu.MMLUBenchmark
    elif name == "hellaswag":
        return hellaswag.HellaSwag
    else:
        raise ValueError(f"Unknown benchmark name: {name}")


@hydra.main(config_path="configs/test/", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Run the evaluation benchmarks."""
    path_base = get_original_cwd()
    model = build_model(ckpt_path=f"{path_base}/{cfg['model_path']}")
    model.eval()
    model.to("cuda")

    # generation hyperparameters
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32' or 'bfloat16' or 'float16'
    device = "cuda"
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    wrapped_model = model_wrapper.ModelWrapper(model, ctx, cfg)

    for benchmark_name in cfg["benchmarks"]:
        benchmark = get_benchmark_class(name=benchmark_name)(
            name=benchmark_name,
            model=wrapped_model,
            cache_dir=path_base + f"/data/eval/{benchmark_name}",
        )
        print(f"{benchmark_name}: {benchmark.execute()}")


if __name__ == "__main__":
    # ignore parameter warning
    # pylint: disable=no-value-for-parameter
    main()

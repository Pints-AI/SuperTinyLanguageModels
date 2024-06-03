"""Tests for each of the benchmarks"""
import pytest

from evals.mcqs.load_benchmarks import load_benchmark

def test_load_benchmark():
    benchmark_names = ["arc", "winograd", "mmlu", "hellaswag", "blimp"]
    for benchmark_name in benchmark_names:
        for split in ["test", "validation"]:
            benchmark = load_benchmark(benchmark_name, split)
            assert benchmark is not None
            for item in benchmark:
                assert len(item) == 3
                assert isinstance(item[0], str)
                assert isinstance(item[1], str)
                assert isinstance(item[2], list)
                assert all(isinstance(option, str) for option in item[2])
                break

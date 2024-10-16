from evals.core import BaseEvaluator, BaseModelWrapper
from typing import Optional, Callable, Dict, Any, List
from tqdm import tqdm
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import warnings
import gc  # Garbage collector


class TeacherTextModelingEvaluator(BaseEvaluator):
    """
    Evaluate a model's capability of generating text by 
    measuring the perplexity a larger teacher model achieves on the 
    generated outputs.
    """

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        model_generator, 
        yield_fn: Callable,
        yield_fn_params: Optional[Dict[str, Any]] = None,
        generator_params: Optional[Dict[str, Any]] = None, 
        chunk_size: Optional[int] = 100,
        teacher_model_names: Optional[List[str]] = [
            "meta-llama/Llama-3.2-1B",
            "tiiuae/falcon-rw-1b",
            "bigscience/bloom-1b1"
        ],
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.model_generator = model_generator
        self.yield_fn = yield_fn(**(yield_fn_params or {}))
        self.generator_params = generator_params
        self.chunk_size = chunk_size
        self.teacher_model_names = teacher_model_names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def _load_teacher_model(self, teacher_model_name: str):
        """
        Load a teacher model and its tokenizer with quantization and no_grad.

        Args:
            teacher_model_name (str): The HuggingFace model identifier.

        Returns:
            model: The loaded and quantized teacher model.
            tokenizer: The corresponding tokenizer.
        """
        try:
            # Configure quantization (e.g., 4-bit) using BitsAndBytes
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # or "fp4" depending on model support
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name,
                quantization_config=quantization_config,
                device_map=self.device, #"auto",  # Automatically place on available devices
                trust_remote_code=True,  # If the model uses custom code
            )

            model.eval()
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False

            return model, tokenizer

        except Exception as e:
            warnings.warn(f"Failed to load teacher model {teacher_model_name}: {e}")
            return None, None

    def _teacher_evaluate_completion(self, teacher_model, tokenizer, model_completion: str):
        """
        Compute the normalized perplexity of the completion using the teacher model.

        Args:
            teacher_model: The loaded teacher model.
            tokenizer: The tokenizer corresponding to the teacher model.
            model_completion (str): The text generated by the model being evaluated.

        Returns:
            float: The normalized perplexity score.
        """
        if teacher_model is None or tokenizer is None:
            return float('inf')  # Assign a high perplexity if model/tokenizer failed to load

        try:
            # Tokenize the completion with appropriate settings
            inputs = tokenizer(
                model_completion,
                return_tensors='pt',
                truncation=True,
                max_length=512, # to minimize vRAM burden 
                padding=False
            ).to(self.device)

            # Ensure input_ids are present
            if 'input_ids' not in inputs:
                return float('inf')

            input_ids = inputs['input_ids']

            with torch.no_grad():
                # Get the logits from the model
                outputs = teacher_model(input_ids, labels=input_ids)
                # CrossEntropyLoss includes a mean over all tokens
                # To get total loss, multiply by number of tokens
                # Then compute perplexity
                loss = outputs.loss
                # Multiply loss by number of tokens to get total loss
                total_loss = loss.item() * input_ids.size(1)
                # Compute perplexity
                perplexity = np.exp(total_loss / input_ids.size(1))
                return perplexity

        except Exception as e:
            warnings.warn(f"Failed to compute perplexity for completion: {e}")
            return float('inf')

    def _cleanup_cuda_memory(self):
        """
        Clean up CUDA memory by deleting unused variables and emptying cache.
        """
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, model):
        """
        Evaluate the model's text modeling capabilities.

        Args:
            model: The model to be evaluated.

        Returns:
            Dict[str, Any]: A dictionary with the evaluation results.
        """
        try:
            # Wrap the model using the provided model wrapper
            model = self.model_wrapper(
                model=model,
                model_generator=self.model_generator,
                generator_params=self.generator_params
            )

            model_completions = []
            for text_prompt in tqdm(self.yield_fn, desc="Generating Text Completions for Teacher Modeling"):
                model_completions.append(model(text_prompt))

            # Iterate over the teacher models, load them and calculate the length-normalized perplexity
            teacher_evaluations = {}
            for teacher_model_name in self.teacher_model_names:
                # Load teacher model and tokenizer
                teacher_model, tokenizer = self._load_teacher_model(teacher_model_name=teacher_model_name)
                if teacher_model is None or tokenizer is None:
                    teacher_evaluations[teacher_model_name] = float('inf')
                    continue

                scores = []
                # Evaluate each completion
                for completion in tqdm(model_completions, desc=f"Evaluating with {teacher_model_name}", leave=False):
                    score = self._teacher_evaluate_completion(
                        teacher_model=teacher_model,
                        tokenizer=tokenizer,
                        model_completion=completion
                    )
                    scores.append(score)

                # Compute the mean perplexity, ignoring infinite scores
                finite_scores = [s for s in scores if np.isfinite(s)]
                if finite_scores:
                    mean_perplexity = np.mean(finite_scores)
                else:
                    mean_perplexity = float('inf')

                teacher_evaluations[teacher_model_name] = mean_perplexity

                # Clean up after each teacher model evaluation
                del teacher_model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            # Compute average normalized perplexity across teacher models
            finite_teacher_scores = [
                score for score in teacher_evaluations.values() if np.isfinite(score)
            ]
            if finite_teacher_scores:
                avg_norm_perplexity = np.mean(finite_teacher_scores)
            else:
                avg_norm_perplexity = float('inf')

            teacher_evaluations["Avg. Norm. Perplexity"] = avg_norm_perplexity

            return {
                "benchmark_type": "Teacher Text Modeling",
                "benchmark_name": self.env_id,
                "results": teacher_evaluations,
            }

        finally:
            # Ensure that CUDA memory is cleaned up even if an error occurs
            self._cleanup_cuda_memory()

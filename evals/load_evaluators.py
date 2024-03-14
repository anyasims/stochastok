"""
Given an evaluator name, load the evaluator
"""

from evals.evaluator_interface import EvaluationInterface
from evals.mcqs.mcq_evaluator import MCQEvaluator
from evals.generation_evaluator_math import GenerationEvaluatorMath

EVALUATORS_DICT = {
    "mcq": MCQEvaluator,
    "math_generation": GenerationEvaluatorMath,
}


def load_evaluator(evaluator_name, model, **kwargs) -> EvaluationInterface:
    """
    Given the evaluator name, load the evaluator
    """
    return EVALUATORS_DICT[evaluator_name](model, **kwargs)

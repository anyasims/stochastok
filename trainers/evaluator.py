"""Code for running samples from the evaluation benchmarks"""

from evals.load_evaluators import load_evaluator

def train_eval(eval_cfg, model, cfg):
    """Train the model"""
    evaluator_name = eval_cfg["evaluator"]
    kwargs = {
        key: value for key, value in eval_cfg.items() if key != "evaluator"
    }
    evaluator = load_evaluator(evaluator_name, model, **kwargs, cfg=cfg)
    results = evaluator.evaluate()
    return results

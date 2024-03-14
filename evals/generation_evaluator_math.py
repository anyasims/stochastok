"""
Evaluator class for evaluating models.
Question-answering datasets.
Compares the likelihoods of the question+correct_answer with the likelihoods of the question+incorrect_answer(s).
"""

import torch
import tqdm
import random
import os
from datasets import DatasetDict

from evals.evaluator_interface import EvaluationInterface


class GenerationEvaluatorMath(EvaluationInterface):
    """
    Base Evaluator class the evaluates models
    and prints/logs the results.
    """

    def __init__(self, model, cfg, num_samples=None, **kwargs):
        self.model = model
        self.num_samples = num_samples
        # make sure the model is in eval model
        self.model.eval()
        self.tokenizer = model.embedding_model.tokenizer

        as_datasets_path = os.path.join(
            cfg["general"]["paths"]["data_dir"],
            "data_as_datasets",
            f"multi_digit_addition",
        )
        self.dataset = DatasetDict.load_from_disk(as_datasets_path)

        max_tokens=20
        temperature=0.0
        top_k=None
        batch_size=min(100, num_samples)
        context_window = cfg["model"]["context_window"]

        self.max_tokens = min(max_tokens, context_window)
        self.temperature = temperature
        self.top_k = top_k
        self.batch_size = batch_size

        print(f"\nGenerationEvaluatorMath initialized.")
        print(f"Datset loaded from: {as_datasets_path}\n")
        print(f"Batch size: {self.batch_size}")


    def evaluate_split(self, split, disable_tqdm=True):
        dataset = self.dataset[split]
        dataset = dataset.shuffle(seed=42)
        num_samples = min(self.num_samples, len(dataset))
        num_batches = len(range(0, num_samples, self.batch_size))
        
        metrics = {}
        for tokenization_type in ["base", "stochastok", "character"]:
            answer_found = []
            for i in tqdm.tqdm(range(0, num_samples, self.batch_size),
                                disable=disable_tqdm,
                                desc=f"Eval for Math Addition ({tokenization_type=}, {split=})",
                                total=num_batches,
                            ):
                batch = dataset[i : i + self.batch_size]
                question_ids = batch[f"t_question_{tokenization_type}"]
                answer_string = batch[f"answer"]
                generations = batch_generate(self.model, question_ids, self.max_tokens, self.temperature, self.top_k)
                contains_answer = [a in g for a, g in zip(answer_string, generations)]
                answer_found.extend(contains_answer)
                if i == 0:
                    print(f"Split: {split}, Tokenization type: {tokenization_type}, Answer found: {contains_answer[0]}, Generation: {generations[0]}, Answer: {answer_string[0]}")
            metrics[f"{tokenization_type}/answer_found"] = sum(answer_found) / len(answer_found)
        return metrics

    def evaluate(self, disable_tqdm=True):
        results = {}
        for split in ["train", "val"]:
            results[f"Addition/{split}"] = self.evaluate_split(split, disable_tqdm=disable_tqdm)
        return results
    

@torch.no_grad()
def batch_generate(model, tokenized_s, max_tokens, temperature=1.0, top_k=None):
    batch_size = len(tokenized_s)
    vocab_size = model.embedding_model.tokenizer.vocab_size
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert device.type == "cuda"
    model.eval()
    token_ids_s_saved, prompt_mask_s = model.embedding_model.tokenizer.pad_batch(tokenized_s, direction="right")
    model = model.to(device)
    token_ids_s_saved = token_ids_s_saved.to(device)
    prompt_mask_s = prompt_mask_s.to(torch.bool).to(device)

    has_eot_token = torch.zeros(batch_size, dtype=torch.bool, device=device)

    min_length = min([len(s) for s in tokenized_s])
    token_ids_s = token_ids_s_saved[:, :min_length]
    max_new_tokens = max_tokens - max([len(s) for s in token_ids_s_saved])
    for _ in range(max_new_tokens):
        logits_s = model(token_ids=token_ids_s, attention_mask=None)[0][:, -1, :]
        assert logits_s.shape == (batch_size, vocab_size)
        if temperature == 0.0:
            token_id_next_s = torch.argmax(logits_s, dim=-1)
        else:
            logits_s = logits_s / temperature
            if top_k is not None:
                logits_s_k, idx_s_k = torch.topk(logits_s, min(top_k, vocab_size), dim=-1) # (batch_size, top_k)
                probs_k = torch.nn.functional.softmax(logits_s_k, dim=-1) # (batch_size, top_k)
                idxs = torch.multinomial(probs_k, num_samples=1) # (batch_size, 1)
                token_id_next_s = idx_s_k[torch.arange(batch_size), idxs] # (batch_size)
            else:
                probs = torch.nn.functional.softmax(logits_s, dim=-1) # (batch_size, vocab_size)
                token_id_next_s = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                
        has_eot_token = torch.bitwise_or(has_eot_token, token_id_next_s == model.embedding_model.tokenizer.eot_token)
        if has_eot_token.all():
            break
        t = token_ids_s.shape[1]
        assert token_id_next_s.shape == (batch_size,)
        if t < token_ids_s_saved.shape[1]:
            token_id_next_s = torch.where(prompt_mask_s[:, t], token_ids_s_saved[:, t], token_id_next_s)
        token_ids_s = torch.cat((token_ids_s, token_id_next_s.unsqueeze(1)), dim=1)
    
    return model.embedding_model.tokenizer.decode_batch(token_ids_s)

    generated_part = [token_ids_s[i][len(tokenized_s[i]):].tolist() for i in range(batch_size)]
    generation_s = model.embedding_model.tokenizer.decode_batch(generated_part)
    return generation_s
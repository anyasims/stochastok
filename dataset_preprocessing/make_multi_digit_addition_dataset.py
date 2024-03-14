import os
import numpy as np
import tqdm
import time
import datasets
from datasets import Dataset, DatasetDict
import itertools

from models.components.base_tokenizer import BaseTokenizer
from stochastok_processor import StochastokProcessor
from dataset_preprocessing.utils import save_as_memmaps

# Utils
def collect_question_answers(t_question_answers, tokenizer):
    all_ids = [[int(x)] for x in set(t_question_answers)]
    all_tokens = tokenizer.decode_batch(all_ids)
    all_tokens.remove("$")
    all_tokens.remove("=")
    assert all(["$" not in s for s in all_tokens])
    assert all(["=" not in s for s in all_tokens])
    dollar_id = tokenizer.encode("$")[0]
    equal_id = tokenizer.encode("=")[0]
    t_questions = []
    t_answers = []
    is_question = True
    prev_i = 0
    for i in tqdm.tqdm(range(1, len(t_question_answers)), total=len(t_question_answers)):
        if is_question:
            assert t_question_answers[i] != dollar_id
            if t_question_answers[i] == equal_id:
                t_questions.append(t_question_answers[prev_i:i+1])
                prev_i = i+1
                is_question = False
        else:
            assert t_question_answers[i] != equal_id
            if t_question_answers[i] == dollar_id or i == len(t_question_answers)-1:
                t_answers.append(t_question_answers[prev_i:i])
                prev_i = i
                is_question = True
    return t_questions, t_answers

def batched_expand(ids, stochastok_processor, batch_length=1000):
    all_expansions = []
    for i in tqdm.tqdm(range(0, len(ids), batch_length), total=len(ids)//batch_length, desc="Batched expanding"):
        ids_batch = ids[i:i+batch_length]
        stochastok_batch = stochastok_processor.expand(ids_batch, 1.0, max_num_to_expand=len(ids_batch), disable_tqdm=True)
        all_expansions.extend(stochastok_batch)
    return all_expansions

# Make questions
def make_questions(num_digits, train_size, val_size, tokenizer, stochastok_processor):
    total_size = train_size + val_size
    all_number_pairs = np.arange(10**(2*num_digits-1), 10**(2*num_digits))
    assert len(all_number_pairs) >= total_size, f"{len(all_number_pairs)} < {total_size}"
    np.random.shuffle(all_number_pairs)
    questions = []
    answers = []
    t_question_answers = []
    t_question_answers_character = []
    for number_pair in tqdm.tqdm(all_number_pairs[:total_size], total=total_size):
        number1 = number_pair // 10**num_digits
        number2 = number_pair % 10**num_digits
        answer = str(number1 + number2)[::-1]
        question = f"${number1}+{number2}="
        questions.append(question)
        answers.append(answer)
        t_question_answers.append(tokenizer.encode(question+answer))
        t_characters = [x[0] for x in tokenizer.encode_batch([c for c in question+answer])]
        t_question_answers_character.append(t_characters)

    train_questions = questions[:train_size]
    train_answers = answers[:train_size]
    val_questions = questions[train_size:]
    val_answers = answers[train_size:]

    train_t_question_answers_flat = list(itertools.chain.from_iterable(t_question_answers[:train_size]))
    val_t_question_answers_flat = list(itertools.chain.from_iterable(t_question_answers[train_size:]))
    train_t_question_answers_character_flat = list(itertools.chain.from_iterable(t_question_answers_character[:train_size]))
    val_t_question_answers_character_flat = list(itertools.chain.from_iterable(t_question_answers_character[train_size:]))

    train_t_question_answers_stochastok_flat = batched_expand(train_t_question_answers_flat, stochastok_processor)
    val_t_question_answers_stochastok_flat = batched_expand(val_t_question_answers_flat, stochastok_processor)

    train_t_question_answers_flat = np.array(train_t_question_answers_flat)
    val_t_question_answers_flat = np.array(val_t_question_answers_flat)
    train_t_question_answers_character_flat = np.array(train_t_question_answers_character_flat)
    val_t_question_answers_character_flat = np.array(val_t_question_answers_character_flat)
    train_t_question_answers_stochastok_flat = np.array(train_t_question_answers_stochastok_flat)
    val_t_question_answers_stochastok_flat = np.array(val_t_question_answers_stochastok_flat)
    
    train_t_questions, train_t_answers = collect_question_answers(train_t_question_answers_flat, tokenizer)
    val_t_questions, val_t_answers = collect_question_answers(val_t_question_answers_flat, tokenizer)
    train_t_questions_character, train_t_answers_character = collect_question_answers(train_t_question_answers_character_flat, tokenizer)
    val_t_questions_character, val_t_answers_character = collect_question_answers(val_t_question_answers_character_flat, tokenizer)
    train_t_questions_stochastok, train_t_answers_stochastok = collect_question_answers(train_t_question_answers_stochastok_flat, tokenizer)
    val_t_questions_stochastok, val_t_answers_stochastok = collect_question_answers(val_t_question_answers_stochastok_flat, tokenizer)
    return (
        # Untokenized
        train_questions, train_answers,
        val_questions, val_answers,
        # Tokenized deterministic
        train_t_questions, train_t_answers,
        val_t_questions, val_t_answers,
        # Tokenized character-wise
        train_t_questions_character, train_t_answers_character,
        val_t_questions_character, val_t_answers_character,
        # Tokenized stochastok
        train_t_questions_stochastok, train_t_answers_stochastok,
        val_t_questions_stochastok, val_t_answers_stochastok,
        # Flattened for memmaps
        train_t_question_answers_flat, val_t_question_answers_flat,
        train_t_question_answers_character_flat, val_t_question_answers_character_flat,
        train_t_question_answers_stochastok_flat, val_t_question_answers_stochastok_flat,
    )

# Initialize tokenizer and stochastok processor
tokenizer = BaseTokenizer()
stochastok_processor = StochastokProcessor(tokenizer=tokenizer.tokenizer)

# Make questions
print("Making questions...")
num_digits = 3
start_time = time.time()
total_numbers = 10**(2*num_digits) - 10**(2*num_digits-1)
val_size = total_numbers // 10
train_size = total_numbers - val_size
dataset_size = train_size + val_size
(
    train_questions, train_answers,
    val_questions, val_answers,
    train_t_questions, train_t_answers,
    val_t_questions, val_t_answers,
    train_t_questions_character, train_t_answers_character,
    val_t_questions_character, val_t_answers_character,
    train_t_questions_stochastok, train_t_answers_stochastok,
    val_t_questions_stochastok, val_t_answers_stochastok,
    train_t_question_answers_flat, val_t_question_answers_flat,
    train_t_question_answers_character_flat, val_t_question_answers_character_flat,
    train_t_question_answers_stochastok_flat, val_t_question_answers_stochastok_flat,
) = make_questions(num_digits, train_size, val_size, tokenizer, stochastok_processor)
print(f"Time taken: {time.time() - start_time} seconds\n")

# Save as datasets
as_datasets_path = os.path.join("./data", "data_as_datasets", "multi_digit_addition")
print(f"Saving as datasets to {as_datasets_path}...")
if not os.path.exists(as_datasets_path):
    os.makedirs(as_datasets_path)
train_dataset = Dataset.from_dict({
    "question": train_questions[:val_size],
    "answer": train_answers[:val_size],
    "t_question_base": train_t_questions[:val_size],
    "t_answer_base": train_t_answers[:val_size],
    "t_question_character": train_t_questions_character[:val_size],
    "t_answer_character": train_t_answers_character[:val_size],
    "t_question_stochastok": train_t_questions_stochastok[:val_size],
    "t_answer_stochastok": train_t_answers_stochastok[:val_size],
})
val_dataset = Dataset.from_dict({
    "question": val_questions,
    "answer": val_answers,
    "t_question_base": val_t_questions,
    "t_answer_base": val_t_answers,
    "t_question_character": val_t_questions_character,
    "t_answer_character": val_t_answers_character,
    "t_question_stochastok": val_t_questions_stochastok,
    "t_answer_stochastok": val_t_answers_stochastok,
})
as_datasets_dataset = DatasetDict({
    "train": train_dataset,
    "val": val_dataset
})
as_datasets_dataset.save_to_disk(as_datasets_path)

# Save as memmaps
as_memmaps_path_base = os.path.join("./data", "data_as_memmaps", "multi_digit_addition_base")
as_memmaps_path_stochastok = os.path.join("./data", "data_as_memmaps", "multi_digit_addition_stochastok")
as_memmaps_path_character = os.path.join("./data", "data_as_memmaps", "multi_digit_addition_character")
print(f"Saving as memmaps to {as_memmaps_path_base}, {as_memmaps_path_stochastok}, {as_memmaps_path_character}...")
for as_memmaps_path, train_data, val_data in zip([as_memmaps_path_base, as_memmaps_path_stochastok, as_memmaps_path_character], [train_t_question_answers_flat, train_t_question_answers_stochastok_flat, train_t_question_answers_character_flat], [val_t_question_answers_flat, val_t_question_answers_stochastok_flat, val_t_question_answers_character_flat]):
    if not os.path.exists(as_memmaps_path):
        os.makedirs(as_memmaps_path)
    save_as_memmaps(train_data, "train", as_memmaps_path)
    save_as_memmaps(val_data, "val", as_memmaps_path)
print(f"Memmaps saved.\n")

# Load dataset and print some examples (as a check)
print("Loading dataset and printing some examples...")
loaded_dataset = datasets.load_from_disk(as_datasets_path)
for split in ["train", "val"]:
    for i in range(3):
        for key in loaded_dataset[split].column_names:
            print(f"{key}: {loaded_dataset[split][key][i]}")
        for tokenization_type in ["base", "character", "stochastok"]:
            assert tokenizer.decode(loaded_dataset[split][f"t_answer_{tokenization_type}"][i]) == loaded_dataset[split]["answer"][i], f"{tokenizer.decode(loaded_dataset[split][f't_answer_{tokenization_type}'][i])} != {loaded_dataset[split]['answer'][i]}"
        print()

# Load memmaps and print some examples (as a check)
print("Loading memmaps and printing some examples...")
loaded_train_data = np.memmap(f"{as_memmaps_path_base}/train.bin", dtype=np.uint16, mode="r")
loaded_train_data_stochastok = np.memmap(f"{as_memmaps_path_stochastok}/train.bin", dtype=np.uint16, mode="r")
loaded_train_data_character = np.memmap(f"{as_memmaps_path_character}/train.bin", dtype=np.uint16, mode="r")
print(f"{loaded_train_data.shape=}, {loaded_train_data_stochastok.shape=}, {loaded_train_data_character.shape=}")
step = 20
for i in range(0, 60, step):
    print(f"Tokens {i}-{i+step}:")
    print(tokenizer.decode(loaded_train_data[i:i+step]))
    print(tokenizer.decode(loaded_train_data_stochastok[i:i+step]))
    print(tokenizer.decode(loaded_train_data_character[i:i+step]))

print(f"\nMulti-digit addition dataset generation complete.")

# run with:
# python dataset_preprocessing/make_multi_digit_addition_dataset.py
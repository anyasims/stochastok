import os
import numpy as np
import random
import tqdm
import time
import datasets
from datasets import Dataset, DatasetDict
import json

from models.components.base_tokenizer import BaseTokenizer
from dataset_preprocessing.utils import save_as_memmaps


# Building blocks
placeholder_options = "<OPTIONS>"
synonyms_options = ["options", "choices", "option words", "option strings"]
placeholder_option = "<OPTION>"
synonyms_option = ["word", "", "string", "option", "choice", "option word", "option string"]
placeholder_the = "<THE>"
synonyms_the = ["the", "the possible", "the available"]
question1_part1_strings = [
    "Which <OPTION>",
    "What <OPTION>",
    "Which of <THE> <OPTIONS>",
]
question1_option_part_strings = [
    "<THE> <OPTIONS>:",
    "<THE> <OPTIONS> are:",
    "These are <THE> <OPTIONS>:",
]
question2_strings = [
    "<THE> <OPTIONS>:",
    "<THE> <OPTIONS> were:",
    "Repeat <THE> <OPTIONS>:",
]
q0_mostofletter_strings = [
    "has the most letter '<AUX>'s?",
]
q1_contains_strings = [
    "contains '<AUX>'?",
]
q2_startswith_strings = [
    "starts with '<AUX>'?",
]
q3_endswith_strings = [
    "ends with '<AUX>'?",
]
q4_longest_strings = [
    "is the longest?",
]
q5_shortest_strings = [
    "is the shortest?",
]
q_strings_list = [
    q0_mostofletter_strings,
    q1_contains_strings,
    q2_startswith_strings,
    q3_endswith_strings,
    q4_longest_strings,
    q5_shortest_strings,
]
q_descriptions = [
    "most of a letter",
    "contains",
    "starts with",
    "ends with",
    "longest",
    "shortest",
]

# Functions for generating each type of question
def q0_mostofletter_aux_and_options(all_words, total_options):
    success = False
    letter = random.choice("abcdefghijklmnopqrstuvwxyz")
    correct_option = None
    incorrect_options = []
    min_for_correct = 2
    max_for_incorrect = 1
    # random permutation of len(all_words)
    idxs = np.random.permutation(len(all_words))
    for idx in idxs:
        num_of_letter = all_words[idx].count(letter)
        if min_for_correct <= num_of_letter and correct_option is None:
            correct_option = all_words[idx]
        elif num_of_letter <= max_for_incorrect and len(incorrect_options) < total_options-1:
            incorrect_options.append(all_words[idx])
        if len(incorrect_options) >= total_options-1 and correct_option is not None:
            success = True
            break
    all_options = [correct_option] + incorrect_options
    return success, letter, all_options

def q1_contains_aux_and_options(all_words, total_options):
    success = False
    word = random.choice(all_words)
    subword_length = random.randint(1, min(3, len(word)))
    start = random.randint(0, len(word)-subword_length)
    subword = word[start:start+subword_length]
    correct_option = word
    incorrect_options = []
    # random permutation of len(all_words)
    idxs = np.random.permutation(len(all_words))
    for idx in idxs:
        if subword not in all_words[idx]:
            incorrect_options.append(all_words[idx])
        if len(incorrect_options) >= total_options-1:
            success = True
            break
    all_options = [correct_option] + incorrect_options
    return success, subword, all_options

def q7_startswith_aux_and_options(all_words, total_options):
    success = False
    word = random.choice(all_words)
    subword_length = random.randint(1, min(3, len(word)))
    subword = word[:subword_length]
    correct_option = word
    incorrect_options = []
    # random permutation of len(all_words)
    idxs = np.random.permutation(len(all_words))
    for idx in idxs:
        if not all_words[idx].startswith(subword):
            incorrect_options.append(all_words[idx])
        if len(incorrect_options) >= total_options-1:
            success = True
            break
    all_options = [correct_option] + incorrect_options
    return success, subword, all_options

def q8_endswith_aux_and_options(all_words, total_options):
    success = False
    word = random.choice(all_words)
    subword_length = random.randint(1, min(3, len(word)))
    subword = word[-subword_length:]
    correct_option = word
    incorrect_options = []
    # random permutation of len(all_words)
    idxs = np.random.permutation(len(all_words))
    for idx in idxs:
        if not all_words[idx].endswith(subword):
            incorrect_options.append(all_words[idx])
        if len(incorrect_options) >= total_options-1:
            success = True
            break
    all_options = [correct_option] + incorrect_options
    return success, subword, all_options

def q4_longest_aux_and_options(all_words, total_options):
    success = False
    correct_option = None
    incorrect_options = []
    min_for_correct = 7
    max_for_incorrect = 4
    # random permutation of len(all_words)
    idxs = np.random.permutation(len(all_words))
    for idx in idxs:
        length = len(all_words[idx])
        if min_for_correct <= length and correct_option is None:
            correct_option = all_words[idx]
        elif length <= max_for_incorrect and len(incorrect_options) < total_options-1:
            incorrect_options.append(all_words[idx])
        if len(incorrect_options) >= total_options-1 and correct_option is not None:
            success = True
            break
    all_options = [correct_option] + incorrect_options
    return success, None, all_options

def q5_shortest_aux_and_options(all_words, total_options):
    success = False
    correct_option = None
    incorrect_options = []
    max_for_correct = 3
    min_for_incorrect = 6
    # random permutation of len(all_words)
    idxs = np.random.permutation(len(all_words))
    for idx in idxs:
        length = len(all_words[idx])
        if length <= max_for_correct and correct_option is None:
            correct_option = all_words[idx]
        elif min_for_incorrect <= length and len(incorrect_options) < total_options-1:
            incorrect_options.append(all_words[idx])
        if len(incorrect_options) >= total_options-1 and correct_option is not None:
            success = True
            break
    all_options = [correct_option] + incorrect_options
    return success, None, all_options

q_functions = [
    q0_mostofletter_aux_and_options,
    q1_contains_aux_and_options,
    q7_startswith_aux_and_options,
    q8_endswith_aux_and_options,
    q4_longest_aux_and_options,
    q5_shortest_aux_and_options,
]

# Build questions
def build_question(q_idx, all_words, total_options):
    ### Sample the question
    # get the aux and options
    success = False
    for i in range(100):
        success, aux, options = q_functions[q_idx-1](all_words, total_options)
        if success:
            break
    if not success:
        print("Failed to build question")
        return None, None, None
    ### Construct question 1 eg. "Which is the longest word? Options: ... Answer: "
    # sample synonyms for the placeholders
    synonym_options1 = random.choice(synonyms_options)
    synonym_option1 = random.choice(synonyms_option)
    synonym_the1 = random.choice(synonyms_the)
    # sample the question parts
    question1_general_part = random.choice(question1_part1_strings)
    question1_specific_part = random.choice(q_strings_list[q_idx-1])
    question1_options_part = random.choice(question1_option_part_strings)
    # replace the placeholders question1_general_part
    question1_general_part = question1_general_part.replace("<OPTIONS>", synonym_options1)
    question1_general_part = question1_general_part.replace("<OPTION>", synonym_option1)
    question1_general_part = question1_general_part.replace("<THE>", synonym_the1)
    question1_general_part = question1_general_part[0].upper() + question1_general_part[1:]
    # replace the placeholders question1_specific_part
    if "<AUX>" in question1_specific_part:
        assert aux is not None
        question1_specific_part = question1_specific_part.replace("<AUX>", aux)
    # join question parts together
    question1_question_part = f"{question1_general_part} {question1_specific_part}"
    # replace the placeholders question1_options_part
    question1_options_part = question1_options_part.replace("<OPTIONS>", synonym_options1)
    question1_options_part = question1_options_part.replace("<OPTION>", synonym_option1)
    question1_options_part = question1_options_part.replace("<THE>", synonym_the1)
    question1_options_part = question1_options_part[0].upper() + question1_options_part[1:]
    # shuffle and append the options
    i = np.random.randint(len(options))
    shuffled_options = options[i:] + options[:i]
    shuffled_options_str = f" [ {', '.join(shuffled_options)}]"
    question1_options_part = question1_options_part + shuffled_options_str + "."
    # join the question and options together
    if np.random.choice([True, False]):
        question1 = f"{question1_question_part} {question1_options_part} Answer:"
    else:
        question1 = f"{question1_options_part} {question1_question_part} Answer:"
    options_ = [f" {option}" for option in options]
    answer1 = options_[0]
    return question1, answer1, options_

# Load top 1k words
top_1k_words_path = os.path.join("./data_other", "top_1k_words")
top_1k_words = []
with open(top_1k_words_path, "r", encoding="utf-8") as f:
    for line in f:
        top_1k_words.append(line.strip())
assert len(top_1k_words) == 1000

# Print some examples
print("\nPrinting some examples:")
for question_type_idx in range(6):
    question1, answer1, options = build_question(question_type_idx, top_1k_words, total_options=4)
    print(f"Question type {question_type_idx}: {q_descriptions[question_type_idx]}")
    print(f"Question: {question1}")
    print(f"Answer: {answer1}")
    print(f"Options: {options}")

# Tokenizer
tokenizer = BaseTokenizer()

# Generate dataset and tokenize
print("Generating dataset...")
total_options = 4
start_time = time.time()
train_size = int(1e6)
val_size = 1000
dataset_size = train_size + val_size
question_s = []
answer_s = []
options_s = []
full_tokenized_s = []
idxs_for_masks_s = []
for i in tqdm.tqdm(range(int(dataset_size*2))):
    # ^ *2 so that collect a total of dataset_size examples even if some iters fail to find a question
    if len(full_tokenized_s) >= dataset_size:
        break
    question_type_idx = np.random.choice([0, 1, 2, 3, 4, 5])
    question, answer, options = build_question(question_type_idx, top_1k_words, total_options)
    if question is not None:
        question_t = tokenizer.encode(question)
        answer_t = tokenizer.encode(answer)
        full_tokenized = question_t + answer_t
        idxs_for_masks = np.array([0, len(question_t), len(question_t)+len(answer_t)])
        question_s.append(question)
        answer_s.append(answer)
        options_s.append(options)
        full_tokenized_s.append(full_tokenized)
        idxs_for_masks_s.append(idxs_for_masks)
print(f"Time taken: {time.time()-start_time:.2f}s\n")

# Pad for saving as memmaps
print("Padding for saving as memmaps...")
max_len = max([len(full_tokenized) for full_tokenized in full_tokenized_s])
padded_full_tokenized_s = np.array([x + [tokenizer.eot_token]*(max_len-len(x)) for x in full_tokenized_s])
idxs_for_masks_s = np.array(idxs_for_masks_s)
padded_full_tokenized_s.shape, idxs_for_masks_s.shape
print(f"Padding done.\n")

# Save as memmaps
as_memmaps_path = os.path.join("./data", "data_as_memmaps", "langgame")
print(f"Saving as memmaps to {as_memmaps_path}...")
if not os.path.exists(as_memmaps_path):
    os.makedirs(as_memmaps_path)
all_tokenize_data = {
    "train": {
        "data": padded_full_tokenized_s[:train_size],
        "idxs_for_masks": idxs_for_masks_s[:train_size],
    },
    "val": {
        "data": padded_full_tokenized_s[train_size:],
        "idxs_for_masks": idxs_for_masks_s[train_size:],
    }
}
for split, split_data in all_tokenize_data.items():
    for name, data in split_data.items():
        save_as_memmaps(data, f"{split}_{name}", as_memmaps_path)
shapes = {"train_size": train_size, "val_size": val_size}
with open(os.path.join(as_memmaps_path, f"shapes.json"), "w") as f:
    json.dump(shapes, f)
print(f"Memmaps saved.\n")

# Load memmaps and print (as a check)
print("Loading memmaps and printing shapes (as a check)...")
# Load the memmap files
loaded_train_data = np.memmap(f"{as_memmaps_path}/train_data.bin", dtype=np.uint16, mode="r").reshape(train_size, -1)
loaded_train_idxs_for_masks = np.memmap(f"{as_memmaps_path}/train_idxs_for_masks.bin", dtype=np.uint16, mode="r").reshape(train_size, 3)
loaded_val_data = np.memmap(f"{as_memmaps_path}/val_data.bin", dtype=np.uint16, mode="r").reshape(val_size, -1)
loaded_val_idxs_for_masks = np.memmap(f"{as_memmaps_path}/val_idxs_for_masks.bin", dtype=np.uint16, mode="r").reshape(val_size, 3)
print(f"{loaded_train_data.shape=}, {loaded_train_idxs_for_masks.shape=}, {loaded_val_data.shape=}, {loaded_val_idxs_for_masks.shape=}\n")

# Save untokenized as dataset
as_datasets_path = os.path.join("./data", "data_as_datasets", "langgame")
print(f"Saving dataset to {as_datasets_path}...")
if not os.path.exists(as_datasets_path):
    os.makedirs(as_datasets_path)
train_dataset = Dataset.from_dict({"question": question_s[:val_size], "options": options_s[:val_size]})
val_dataset = Dataset.from_dict({"question": question_s[train_size:], "options": options_s[train_size:]})
datasets_untokenized = DatasetDict({
    "train": train_dataset,
    "val": val_dataset
})
datasets_untokenized.save_to_disk(as_datasets_path)
print(f"Dataset saved.\n")

# Load and print (as a check)
print("Loading and printing some examples:")
loaded_dataset = datasets.load_from_disk(as_datasets_path)
print(loaded_dataset)
for split in ["train", "val"]:
    for i in range(5):
        print(loaded_dataset[split][i])

print(f"\nLangGame dataset generation complete.")



# run with:
# python dataset_preprocessing/make_langgame_dataset.py
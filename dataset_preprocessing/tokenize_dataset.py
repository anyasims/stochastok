"""
Necessary to be run before training to make sure all of the data is preprcessed etc.
"""
import os 
from functools import partial
import argparse
from datasets import load_dataset

from models.components.base_tokenizer import BaseTokenizer
from dataset_preprocessing.utils import write_tokenized_data_as_memmap


def tokenize(example, tokenizer):
    ids = tokenizer.encode(example["text"])
    ids.append(tokenizer.eot_token)
    return {"ids": ids, "len": len(ids)}
    

def prepare_data_tokenize(
        hf_dataset_name,
        data_dir,
        get_pretokenized_from_hf=False,
        save_to_hf=True,
        hf_username=None,
        ):
    dataset_name = f"{hf_dataset_name.split('/')[-1]}-tokenized"
    tokenized_as_memmap_folder = os.path.join(
        data_dir,
        "data_as_memmaps",
        dataset_name,
    )
    tokenized_memmap_exists = os.path.exists(tokenized_as_memmap_folder) and len(os.listdir(tokenized_as_memmap_folder))!=0
    if tokenized_memmap_exists:
        print(f"Tokenized memmap data already exists (path={tokenized_as_memmap_folder})")
        return
    if not os.path.exists(tokenized_as_memmap_folder):
        os.makedirs(tokenized_as_memmap_folder)


    if get_pretokenized_from_hf:
        if hf_dataset_name == "Skylion007/openwebtext":
            tokenized_hf_id = "anyasims/openwebtext-tokenized"
        else:
            raise ValueError(f"Pre-tokenized HF dataset for {hf_dataset_name} not found.")
        print(f"Loading pretokenized dataset from HuggingFace...")
        tokenized_dataset = load_dataset(tokenized_hf_id)
        print(f"{tokenized_dataset=}")
        assert "ids" in tokenized_dataset["train"].features, "Dataset must contain 'ids' column"
        print(f'{tokenized_dataset["train"][0]["ids"][:10]=}')
        print(f"Successfully loaded pretokenized dataset {tokenized_hf_id} from HuggingFace")
        print(f"Saving to memmap...")
        write_tokenized_data_as_memmap(
            tokenized=tokenized_dataset, 
            tokenized_data_folder=tokenized_as_memmap_folder,
        )
        print(f"Successfully saved flattened tokenized dataset as memmap bin files to {tokenized_as_memmap_folder}")
        successfully_saved_memmap = True
    else:
        dataset_name = dataset_name.split("/")[-1]
        print(f"Loading {hf_dataset_name} dataset from HuggingFace...")
        dataset = load_dataset(hf_dataset_name)
        dataset = dataset["train"].train_test_split(
            test_size=0.01, seed=489, shuffle=True
        )
        dataset["val"] = dataset.pop("test")
        print(f"Dataset: {dataset}")
        print(f"Dataset example: {dataset['train'][0]}\n")
        print(f"Tokenizing dataset...")
        tokenizer = BaseTokenizer()
        tokenize_fn = partial(tokenize, tokenizer=tokenizer)
        # Get the maximum number of processors
        max_procs = os.cpu_count() // 4
        # cap at 12 to reduce memory usage
        max_procs = min(max_procs, 12) # Adjust for memory usage
        print(f"Using {max_procs} processors. Can be increased to up to {os.cpu_count()=}.")
        # tokenize the dataset
        dataset_tokenized = dataset.map(
            tokenize_fn,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=max_procs
        )
        successfully_tokenized = True

        if successfully_tokenized:
            # wrap in try such that half-complete files can be deleted on error
            try:
                # save as memmap bin files (concatenate all the ids in each dataset)
                write_tokenized_data_as_memmap(
                    tokenized=dataset_tokenized, 
                    tokenized_data_folder=tokenized_as_memmap_folder,
                )
                print(f"Successfully saved flattened tokenized dataset as memmap bin files to {tokenized_as_memmap_folder}")
                successfully_saved_memmap = True
            except Exception as exc:
                print(f"Error: {exc}")
                for file in os.listdir(tokenized_as_memmap_folder):
                    os.remove(os.path.join(tokenized_as_memmap_folder, file))
                raise RuntimeError("Failed to process and write data") from exc

            if save_to_hf:
                hf_repo_id = f"{hf_username}/{dataset_name}"
                try:
                    print(f"Attempting to push unflattened dataset to: {hf_repo_id}")
                    dataset_tokenized.push_to_hub(hf_repo_id)
                    print(f"Successfully pushed unflattened dataset to: https://huggingface.co/datasets/{hf_repo_id}")
                    successfully_pushed_to_hf = True
                except Exception as e:
                    print(f"Pushing to HuggingFace failed: {e}")
                    try:
                        print(f"Attempting to save to local directory")
                        tokenized_as_datasets_folder = os.path.join(
                            data_dir,
                            "data_as_datasets",
                            dataset_name,
                        )
                        if not os.path.exists(tokenized_as_datasets_folder):
                            os.makedirs(tokenized_as_datasets_folder)
                        dataset_tokenized.save_to_disk(tokenized_as_datasets_folder)
                        print(f"Successfully saved unflattened dataset to: {tokenized_as_datasets_folder}")
                    except Exception as e:
                        print(f"Saving unflattened dataset to local directory failed: {e}")

        print(f"\n\nSuccessfully prepared dataset for training: {successfully_saved_memmap}")
        print(f"Dataset path: {tokenized_as_memmap_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", default="Skylion007/openwebtext", type=str)
    parser.add_argument("--data_dir", default="./data", type=str, help="Path to the data directory")
    parser.add_argument("--get_pretokenized_from_hf", default=False, type=bool, help="Load a dataset that is already tokenized from HuggingFace")
    parser.add_argument("--save_to_hf", default=False, type=bool, help="Save to HuggingFace after tokenization.")
    parser.add_argument("--hf_username", default=None, type=str, help="HuggingFace username used if save-to-hf==True.")
    args = parser.parse_args()
    print(f"\nArgs: {args}\n")
    if args.save_to_hf:
        assert args.hf_username is not None, "hf_username must be provided if args.save_to_hf==True"

    prepare_data_tokenize(
        hf_dataset_name=args.hf_dataset_name,
        data_dir=args.data_dir,
        get_pretokenized_from_hf=args.get_pretokenized_from_hf,
        save_to_hf=args.save_to_hf,
        hf_username=args.hf_username,
    )

    # run with:
    # python dataset_preprocessing/tokenize_dataset.py --hf_dataset_name Skylion007/openwebtext --data_dir ./data --get_pretokenized_from_hf True --save_to_hf True --hf_username XXX
    # python dataset_preprocessing/tokenize_dataset.py --get_pretokenized_from_hf True --hf_username anyasims
    # python dataset_preprocessing/tokenize_dataset.py --save_to_hf True --hf_username anyasims
    # python dataset_preprocessing/tokenize_dataset.py --get_pretokenized_from_hf True
            

    




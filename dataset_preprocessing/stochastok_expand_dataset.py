"""
Necessary to be run before training to make sure all of the data is preprcessed etc.
"""
import os 
from functools import partial
import argparse
from datasets import load_dataset

from models.components.base_tokenizer import BaseTokenizer
from stochastok_processor import StochastokProcessor
from dataset_preprocessing.utils import write_tokenized_data_as_memmap


def stochastok_expand(example, stochastok_processor):
    ids = stochastok_processor.expand(example["ids"])
    return {"ids": ids, "len": len(ids)}

def prepare_data_expand(
        hf_dataset_name,
        data_dir,
        get_preexpanded_from_hf=False,
        save_to_hf=True,
        hf_username=None,
        expand_prop=0.1,
        ):
    dataset_name = hf_dataset_name.split("/")[-1]
    expanded_dataset_name = f"{dataset_name}-stochastok{expand_prop}"
    expanded_as_memmap_folder = os.path.join(
        data_dir,
        "data_as_memmaps",
        expanded_dataset_name,
    )
    expanded_memmap_exists = os.path.exists(expanded_as_memmap_folder) and len(os.listdir(expanded_as_memmap_folder))!=0
    if expanded_memmap_exists:
        print(f"Stochastok-expanded memmap data already exists (path={expanded_as_memmap_folder})")
        return
    if not os.path.exists(expanded_as_memmap_folder):
        os.makedirs(expanded_as_memmap_folder)

    if get_preexpanded_from_hf:
        hf_expanded_dataset_name = f"{hf_dataset_name}-stochastok{expand_prop}"
        print(f"Loading pretokenized stochastok-expanded dataset {hf_expanded_dataset_name} from HuggingFace...")
        expanded_dataset = load_dataset(hf_expanded_dataset_name)
        print(f"{expanded_dataset=}")
        assert "ids" in expanded_dataset["train"].features, "Dataset must contain 'ids' column"
        print(f'{expanded_dataset["train"][0]["ids"][:10]=}')
        print(f"Successfully loaded stochastok-expanded dataset {hf_expanded_dataset_name} from HuggingFace")
        print(f"Saving to memmap...")
        write_tokenized_data_as_memmap(
            tokenized=expanded_dataset, 
            tokenized_data_folder=expanded_as_memmap_folder,
        )
        print(f"Successfully saved flattened stochastok-expanded dataset as memmap bin files to {expanded_as_memmap_folder}")
        successfully_saved_memmap = True
    else:
        print(f"Tokenizing dataset...")
        tokenized_dataset = load_dataset(hf_dataset_name)
        print(f"{tokenized_dataset=}")
        assert "ids" in tokenized_dataset["train"].features, "Dataset must contain 'ids' column"
        print(f'{tokenized_dataset["train"][0]["ids"][:10]=}')
        print(f"Successfully loaded pretokenized dataset {hf_dataset_name} from HuggingFace")
        print(f"Saving to memmap...")
        tokenizer = BaseTokenizer()
        stochastok_processor = StochastokProcessor(tokenizer=tokenizer.tokenizer, expand_prop=expand_prop)
        expand_fn = partial(stochastok_expand, stochastok_processor=stochastok_processor)
        # wrap in try such that half-complete files can be deleted on error
        try:
            # Get the maximum number of processors
            max_procs = os.cpu_count() // 4
            # cap at 12 to reduce memory usage
            max_procs = min(max_procs, 12) # Adjust for memory usage
            print(f"Using {max_procs} processors. Can be increased to up to {os.cpu_count()=}.")
            # tokenize the dataset
            dataset_tokenized = tokenized_dataset.map(
                expand_fn,
                desc="Stochastok expanding dataset",
                num_proc=max_procs
            )
            successfully_stochastok_expanded = True
            # save as memmap bin files (concatenate all the ids in each dataset)
            write_tokenized_data_as_memmap(
                tokenized=dataset_tokenized, 
                tokenized_data_folder=expanded_as_memmap_folder,
            )
            print(f"Successfully saved flattened stochastok-expanded tokenized dataset as memmap bin files to {expanded_as_memmap_folder}")
            successfully_saved_memmap = True
            
        except Exception as exc:
            print(f"Error: {exc}")
            for file in os.listdir(expanded_as_memmap_folder):
                os.remove(os.path.join(expanded_as_memmap_folder, file))
            raise RuntimeError("Failed to process and write data") from exc

        if successfully_stochastok_expanded:
            if save_to_hf:
                hf_repo_id = f"{hf_username}/{expanded_dataset_name}"
                try:
                    print(f"Attempting to push unflattened stochastok-expanded dataset to: {hf_repo_id}")
                    dataset_tokenized.push_to_hub(hf_repo_id)
                    print(f"Successfully pushed unflattened stochastok-expanded dataset to: https://huggingface.co/datasets/{hf_repo_id}")
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
                        print(f"Successfully saved unflattened stochastok-expanded dataset to: {tokenized_as_datasets_folder}")
                    except Exception as e:
                        print(f"Saving unflattened stochastok-expanded dataset to local directory failed: {e}")

        print(f"\n\nSuccessfully prepared stochastok-expanded dataset for training: {successfully_saved_memmap}")
        print(f"Dataset path: {expanded_as_memmap_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset-name", default="anyasims/openwebtext-tokenized", type=str, required=False)
    parser.add_argument("--data_dir", default="./data", type=str, required=False, help="Path to the data directory")
    parser.add_argument("--get_preexpanded_from_hf", default=False, type=bool, required=False, help="Load a dataset that is already tokenized from HuggingFace")
    parser.add_argument("--save_to_hf", default=False, type=bool, required=False, help="Save to HuggingFace after tokenization.")
    parser.add_argument("--hf_username", default=None, type=str, required=False, help="HuggingFace username used if save-to-hf==True.")
    parser.add_argument("--expand_prop", default=0.1, type=float, required=False, help="Stochastok hyperparameter.")
    args = parser.parse_args()
    print(f"\nArgs: {args}\n")
    if args.save_to_hf:
        assert args.hf_username is not None, "hf_username must be provided if args.save_to_hf==True"

    prepare_data_expand(
        hf_dataset_name=args.hf_dataset_name,
        data_dir=args.data_dir,
        get_preexpanded_from_hf=args.get_preexpanded_from_hf,
        save_to_hf=args.save_to_hf,
        hf_username=args.hf_username,
        expand_prop=args.expand_prop,
    )

    # run with:
    # python dataset_preprocessing/stochastok_expand_dataset.py --hf_dataset_name Skylion007/openwebtext --data_dir ./data --get_preexpanded_from_hf True --save_to_hf True --hf_username XXX --expand_prop 0.1
    # python dataset_preprocessing/stochastok_expand_dataset.py --get_preexpanded_from_hf True --hf_username anyasims
    # python dataset_preprocessing/stochastok_expand_dataset.py --save_to_hf True --hf_username anyasims
    # python dataset_preprocessing/stochastok_expand_dataset.py --get_preexpanded_from_hf True

            

    




###########################################################################
#
#            Author:   Felix Brei, Daniel Gerber
#            Last mod: 06.03.2024
#
# This script iterates over a list of model checkpoints, then trains each
# model for a user defined number of epochs on a chosen dataset (lcquad,
# coypu or orga). The script pauses every five epochs to let the finetuned
# model translate natural language questions into SPARQL queries. The
# results of this generation are then stored under ./results
#
# There is no further evaluation done here, just the question, generated
# SPARQL and expected SPARQL are saved. Take a look at ./eval.py to see how
# the data is evaluated.
#
###########################################################################


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import json
import requests
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
from pathlib import Path
import argparse


def qald_converter(qald_dataset_url):
    r = requests.get(qald_dataset_url).json()["questions"]
    ds = {
        "question": [],
        "query": []
    }

    for q in r:
        english_questions = list(filter(lambda x: x["language"] == "en", q["question"]))
        if len(english_questions) == 0:
            continue
        else:
            question = english_questions[0]["string"]
            ds["question"].append(question)
            ds["query"].append(q["query"]["sparql"])
    
    return Dataset.from_dict(ds)


parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs (default 50)")
parser.add_argument("--dataset", type=str, required=True, choices=["coypu", "orga", "lcquad", "qald10"])
parser.add_argument("--run-id", type=str, required=False, default="", help="This will be appended to the name of the json file, useful for consecutive runs of the script")
parser.add_argument("--force-new-model", action="store_true", help="If true, the script will ignore any pretrained models on the disk and always instantiate a new one")
parser.add_argument("--shuffle-dataset", action="store_true", help="If true, the script will shuffle the dataset before training")
parser.add_argument("--shuffle-seed", type=int, required=False, default=42, help="Seed for the random shuffle")

cmd_args = parser.parse_args()
run_id = cmd_args.run_id
if run_id != "":
    run_id += "_"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = [
#    # T5 family
#    "t5-small",
#    "t5-base",
#    "t5-large",
#    "google/flan-t5-small",
#    "google/flan-t5-base",
#   
#    # BART family
#    "facebook/bart-base",
#    "facebook/bart-large",
#    "facebook/mbart-large-50",
#    "Babelscape/mrebel-base",
#    "Babelscape/mrebel-large",
#
#    # M2M100 family
    "facebook/m2m100_418M",
#    "facebook/nllb-200-distilled-600M"
]

for checkpoint in models:

    dir_prefix = checkpoint.replace("/", "_")
    model_dir = f"models/{dir_prefix}_text2sparql"
    print(f"Running... models/{dir_prefix}_text2sparql")

    if cmd_args.force_new_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        print("Initializing a new instance of the model.")
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            print("Loading model from disk.")
        except:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            print("Initializing a new instance of the model.")

    dataset = cmd_args.dataset
    if dataset == "lc_quad":
        dataset = load_dataset("lc_quad")

        train_ds = dataset["train"].map(lambda row: { "query": row["sparql_dbpedia18"] }).filter(lambda row: row["question"] is not None).select(range(0,19000,100))
        test_ds = dataset["test"].map(lambda row: { "query": row["sparql_dbpedia18"] }).filter(lambda row: row["question"] is not None).select(range(0,4000,100))
    elif dataset == "qald10":
        train_ds = qald_converter("https://raw.githubusercontent.com/KGQA/QALD-10/main/data/qald_9_plus/qald_9_plus_train_wikidata.json")        
        test_ds = qald_converter("https://raw.githubusercontent.com/KGQA/QALD-10/main/data/qald_10/qald_10.json")

    else:
        with open(f"datasets/train_split_{dataset}.json", "r") as fp:
            train_ds = json.load(fp)

        with open(f"datasets/test_split_{dataset}.json", "r") as fp:
            test_ds = json.load(fp)

        train_ds = Dataset.from_dict(train_ds)
        test_ds = Dataset.from_dict(test_ds)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if "t5" in checkpoint:
        prefix = "Translate to SPARQL: "
    else:
        prefix = ""

    max_input_length = 128
    def preprocess(examples):
        model_inputs = tokenizer(prefix + examples["question"], max_length=max_input_length, truncation=True, padding=True)
        model_targets = tokenizer(prefix + examples["query"], max_length=max_input_length, truncation=True, padding=True)

        model_inputs["labels"] = model_targets["input_ids"]

        return model_inputs

    col_names = train_ds.column_names
    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    if cmd_args.shuffle_dataset:
        train_ds = train_ds.shuffle(seed=cmd_args.shuffle_seed)


    for idx in range((cmd_args.num_epochs // 5)):

        args = Seq2SeqTrainingArguments(
            model_dir,
            num_train_epochs = 5,
            evaluation_strategy="epoch",
            save_strategy = "no"
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer)

        trainer = Seq2SeqTrainer(
            args = args,
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_ds,
            eval_dataset = test_ds,
            data_collator = data_collator
        )

        trainer.train()

        results = []
        for item in test_ds:
            inputs =  tokenizer(item["question"], return_tensors="pt").to(device)
            out = tokenizer.batch_decode(model.generate(**inputs, max_new_tokens=256), skip_special_tokens=True)
            results.append({
                "question": item["question"],
                "query": item["query"],
                "generated":  out[0]
            })

        Path("results").mkdir(parents=True, exist_ok=True)
        filename = f"results/{dir_prefix}_{dataset}_{run_id}{idx+1}.json"
        with open(filename, "w") as fp:
            print(f"Writing file: {filename}")
            json.dump(results, fp, indent=4)

    trainer.save_model(model_dir)
    trainer.create_model_card()

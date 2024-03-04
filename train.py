###########################################################################
#
#            Author:   Felix Brei
#            Last mod: 22.02.2024
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
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs (default 50)")
parser.add_argument("--dataset", type=str, required=True, choices=["coypu", "orga", "lcquad"])

cmd_args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = [
    # T5 family
    "t5-small",
    "t5-base",
    "t5-large",
    "google/flan-t5-small",
    "google/flan-t5-base",
   
    # BART family
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/mbart-large-50",
    "Babelscape/mrebel-base",
    "Babelscape/mrebel-large",

    # M2M100 family
    "facebook/m2m100_418M",
    "facebook/nllb-200-distilled-600M"
]

for checkpoint in models:

    dir_prefix = checkpoint.replace("/", "_")
    model_dir = f"models/{dir_prefix}_text2sparql"
    print(f"Running... models/{dir_prefix}_text2sparql")

    dataset = cmd_args.dataset
    if dataset == "lc_quad":
        dataset = load_dataset("lc_quad")

        train_ds = dataset["train"].map(lambda row: { "query": row["sparql_dbpedia18"] }).filter(lambda row: row["question"] is not None).select(range(0,19000,100))
        test_ds = dataset["test"].map(lambda row: { "query": row["sparql_dbpedia18"] }).filter(lambda row: row["question"] is not None).select(range(0,4000,100))
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

    for idx in range((cmd_args.num_epochs // 5)):

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            print("Loading model from disk.")
        except:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            print("Initializing a new instance of the model.")

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
        trainer.save_model(model_dir)
        trainer.create_model_card()

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
        with open(f"results/{dir_prefix}_{dataset}_{idx+1}.json", "w") as fp:
            print(f"Writing file: results/{dir_prefix}_{dataset}_{idx+1}.json")
            json.dump(results, fp, indent=4)


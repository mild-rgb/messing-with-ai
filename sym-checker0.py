from datasets import load_dataset
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, pipeline

raw_datasets = load_dataset("glue", "mrpc")
sym_set = load_dataset("csv", data_files = "synonym-file.csv")


checkpoint = "bert-base-uncased"
tokenizer_sym = AutoTokenizer.from_pretrained(checkpoint)
sym_checker_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

def tokenize_function(example):
	return tokenizer_sym(example["sentence1"], example["sentence2"], truncation = True)
	
tokenized_datasets = raw_datasets.map(tokenize_function, batched = True)
tok_sym_set = sym_set.map(tokenize_function, batched = True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer_sym)


training_args = TrainingArguments("test-trainer")

trainer = Trainer(sym_checker_model, training_args, train_dataset = tokenized_datasets["train"], eval_dataset = tokenized_datasets["validation"], data_collator = data_collator, tokenizer = tokenizer_sym,
)

trainer.train()

print("training 1 finished")

Trainer(sym_checker_model, training_args, train_dataset = tok_sym_set["train"], data_collator = data_collator, tokenizer = tokenizer_sym,
)
trainer.train()

classifier = pipeline("sentiment-analysis", model=sym_checker_model, tokenizer = tokenizer_sym, device = 0)

classifier("testing, testing" "testing, testing")
classifier.save_pretrained("transformers-course")



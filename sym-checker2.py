from datasets import load_dataset
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, pipeline


sym_set = load_dataset("csv", data_files = "identical-meaning-dataset-2.csv")


checkpoint = "bert-base-uncased"
tokenizer_sym = AutoTokenizer.from_pretrained(checkpoint)
sym_checker_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

def tokenize_function(example):
	return tokenizer_sym(example["sentence1"], example["sentence2"], truncation = True)
	
tok_sym_set = sym_set.map(tokenize_function, batched = True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer_sym)


training_args = TrainingArguments("test-trainer")

trainer = Trainer(sym_checker_model, training_args, train_dataset = tok_sym_set["train"], data_collator = data_collator, tokenizer = tokenizer_sym,
)
trainer.train()

classifier1 = pipeline("sentiment-analysis", model=sym_checker_model, tokenizer = tokenizer_sym, device = 0)

classifier1("testing, testing" "testing, testing")
classifier1.save_pretrained("transformers-course/classifier1")

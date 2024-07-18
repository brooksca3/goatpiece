from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EvalPrediction, logging
from datasets import Dataset
import json
import torch
from iteration_tools import wordpiece_perm_generator, get_tokenizer
from data_processing import sep_sents

# disable_caching()
# logging.set_verbosity_error()

# def compute_metrics(eval_results):
#     perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
#     return {"perplexity": perplexity.item()}
def compute_metrics(eval_results):
    return {"average_neg_log_likelihood": eval_results['eval_loss']}

def get_total_tokens(encodings, tokenizer):
    PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PAD]")
    return sum(sum(1 for token_id in sequence if token_id != PAD_TOKEN_ID) for sequence in encodings["input_ids"])


def calculate_perplexity_with_gpt2_loss(tokenizer, training_data, test_data, config=None, batch_size=32, epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
# Customize pre-tokenization and decoding

    torch.cuda.empty_cache()

    train_encodings = tokenizer(training_data, truncation=True, padding=True, add_special_tokens=True, max_length=256)
    test_encodings = tokenizer(test_data, truncation=True, padding=True, add_special_tokens=True, max_length=256)
    # Create Dataset object

    train_dataset = Dataset.from_dict({k: v for k, v in train_encodings.items()})
    test_dataset = Dataset.from_dict({k: v for k, v in test_encodings.items()})

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    if not config:
    # Initialize GPT-2 model
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=256,
            n_ctx=256,
            n_embd=768,
            n_layer=4,
            n_head=4,
            n_inner = 768
        )

    model = GPT2LMHeadModel(config)

    # num_parameters = sum(p.numel() for p in model.parameters())

    # print(f"The GPT-2 model has {num_parameters} parameters.")
    
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_logs",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=1_000_000,
        save_total_limit=0,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()
    # Save the model
# Evaluate the model
    eval_results = trainer.evaluate()
    print(eval_results)
    final_metrics = compute_metrics(eval_results)

    ## clear out the memory explicitly
    del model
    del trainer
    torch.cuda.empty_cache()

    total_tokens = get_total_tokens(test_encodings, tokenizer)
    print('total tokens: ' + str(total_tokens))

    return -1 * final_metrics['average_neg_log_likelihood'] * total_tokens

def get_model(tokenizer, training_data, config=None, batch_size=16, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
# Customize pre-tokenization and decoding

    torch.cuda.empty_cache()

    train_encodings = tokenizer(training_data, truncation=True, padding=True, add_special_tokens=True, max_length=256)
    # Create Dataset object
    train_dataset = Dataset.from_dict({k: v for k, v in train_encodings.items()})

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    if not config:
    # Initialize GPT-2 model
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=256,
            n_ctx=256,
            n_embd=768,
            n_layer=3,
            n_head=3,
            n_inner=768
        )

    model = GPT2LMHeadModel(config)
    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"The GPT-2 model has {num_parameters} parameters.")

    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_logs",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=1_000_000,
        save_total_limit=0,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    torch.cuda.empty_cache()

    return model
        
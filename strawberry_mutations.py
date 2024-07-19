from collections import defaultdict
import torch
import time
import numpy as np
import random
from iteration_tools import wordpiece_perm_generator, get_tokenizer, custom_sort_and_count
from model_playground import get_model
from transformers import BertForMaskedLM, BertTokenizer


def tokens_to_remove_trie_freq(sampled_words, trie, num_to_remove, not_to_touch=[]):
    frequencies = {}
    for word in sampled_words:
        if word in not_to_touch:
            continue
        node = trie.get_node_for_string(word)
        if node:
            frequencies[word] = node.count ** (len(word) / 3) if not word.startswith('##') else node.count ** ((len(word) - 2) / 3)
            # frequencies[word] = node.count ** 1 if not word.startswith('##') else node.count ** 1
        else:
            frequencies[word] = 0

    # Calculate inverse frequencies and normalize them
    inverse_frequencies = {word: 1 / (freq + 1) for word, freq in frequencies.items()}  # Adding 1 to avoid division by zero
    total_inverse_freq = sum(inverse_frequencies.values())

    # Ensure the words are in the same order for numpy choice
    words, normalized_weights = zip(*[(word, inverse_frequencies[word]) for word in sampled_words if word in inverse_frequencies])

    # Normalize weights and use numpy.random.choice to select words without replacement
    normalized_weights = [n / total_inverse_freq for n in normalized_weights]
    words_to_remove = np.random.choice(words, size=num_to_remove, replace=False, p=normalized_weights)

    return list(words_to_remove)


def tokens_to_remove_similar(sampled_words, num_to_remove, trie, not_to_touch=[]):
    word_set = set(sampled_words)
    pairs = {}

    # Find pairs by checking truncated versions of each word
    for word in sampled_words:
        if len(word) > 3 and word not in not_to_touch:
            effective_length = len(word) - 2 if word.startswith("##") else len(word)
            for i in range(1, 3):  # Check for word[:-1], word[:-2], and word[:-3]
                truncated = word[:-i]
                if effective_length - i < 4:
                    break
                if truncated in word_set:
                    truncated_effective_length = len(truncated) - 2 if truncated.startswith("##") else len(truncated)
                    score = min(effective_length, truncated_effective_length)
                    longer_word = max(word, truncated, key=lambda x: len(x) - 2 if x.startswith("##") else len(x))
                    pairs[longer_word] = pairs.get(longer_word, 0) + score
    for key in pairs.keys():
        # Get frequency count from the trie
        node = trie.get_node_for_string(key)
        frequency = node.count if node else 0
        pairs[key] = (pairs[key] ** 4) / (frequency)
    # print(pairs)
    # Normalize the scores
    total_score = sum(pairs.values())
    normalized_weights = [score / total_score for score in pairs.values()]

    # Weighted random choice of words
    words, weights = zip(*pairs.items())
    words_to_remove = np.random.choice(words, size=len(words), replace=False, p=normalized_weights)

    return list(words_to_remove)[:num_to_remove]


def count_specific_token_frequencies(corpus, tokenizer, tokens_of_interest):
    # Initialize frequency dictionary with tokens of interest set to 0
    frequency_dict = {token: 0 for token in tokens_of_interest}

    # Tokenize the corpus
    tokens = tokenizer.tokenize(corpus)

    # Count the frequencies for the tokens of interest
    for token in tokens:
        if token in frequency_dict:
            frequency_dict[token] += 1

    return frequency_dict

def tokens_to_remove_corpus_freq(num_to_remove, corpus, tokenizer, tokens_of_interest, smoothing_val=0.1):
    frequency_dict = count_specific_token_frequencies(corpus, tokenizer, tokens_of_interest)
    
    # Apply smoothing directly to the frequency dictionary
    for token in frequency_dict:
        frequency_dict[token] = 1 / (frequency_dict[token] + smoothing_val)
    # Calculate total after smoothing
    total = sum(frequency_dict.values())

    # Calculate probabilities
    probabilities = [freq / total for freq in frequency_dict.values()]

    # Select tokens based on weighted probabilities
    tokens_to_remove = np.random.choice(list(frequency_dict.keys()), size=num_to_remove, replace=False, p=probabilities)

    return list(tokens_to_remove)

# Example usage
# ... (assuming you have defined corpus, tokenizer, and tokens_of_interest)


def filter_and_sort_words(word_likelihoods, word_frequencies, frequency_threshold, tokenizer, normalized=False):
    # Filter out words that consist of only one token and have a frequency above the threshold
    filtered_words = {
        word: prob for word, prob in word_likelihoods.items()
        if word_frequencies.get(word, 0) >= frequency_threshold
    }

    # Sort the words by their likelihood values from smallest to largest
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1]) if not normalized else sorted(filtered_words.items(), key=lambda x: x[1] / word_frequencies.get(x[0], 1))

    return sorted_words

# def compute_pair_log_likelihoods(test_data, model, tokenizer):
#     pair_likelihoods = defaultdict(float)
#     pair_freqs = defaultdict(int)

#     # Tokenize the test data and return PyTorch tensors
#     test_encodings = tokenizer(test_data, truncation=True, padding=True, add_special_tokens=True, max_length=256, return_tensors='pt')

#     # Move encodings to the same device as the model
#     input_ids = test_encodings['input_ids'].to(model.device)
#     attention_mask = test_encodings['attention_mask'].to(model.device)

#     with torch.no_grad():
#         for i in range(input_ids.size(0)):
#             outputs = model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
#             log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

#             tokenized_test = tokenizer.convert_ids_to_tokens(input_ids[i])
#             last_token = None

#             for j, token_id in enumerate(input_ids[i]):
#                 if token_id == tokenizer.pad_token_id:
#                     break  # Stop processing at the first PAD token

#                 token = tokenized_test[j]
#                 token_log_prob = log_probs[0, j, token_id]

#                 # Check if the current token is a suffix token
#                 if token.startswith("##"):
#                     if last_token is not None:
#                         # Combine the last token (suffix or non-suffix) and current suffix token
#                         token_pair = (last_token, token[2:])  # Remove '##' from the suffix token
#                         pair_likelihoods[token_pair] += token_log_prob.item()
#                         pair_freqs[token_pair] += 1

#                 # Always update the last token, regardless of whether it's a suffix or non-suffix
#                 last_token = token

#     return pair_likelihoods, pair_freqs
def compute_pair_log_likelihoods(test_data, model, tokenizer, batch_size=1024):
    pair_likelihoods = defaultdict(float)
    pair_freqs = defaultdict(int)

    # Tokenize the test data and return PyTorch tensors
    test_encodings = tokenizer(test_data, truncation=True, padding=True, add_special_tokens=True, max_length=256, return_tensors='pt')

    # Move encodings to the same device as the model
    input_ids = test_encodings['input_ids'].to(model.device)
    attention_mask = test_encodings['attention_mask'].to(model.device)

    # Calculate the number of batches
    total_examples = input_ids.size(0)
    num_batches = (total_examples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Compute start and end indices for the current batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_examples)

            # Process the batch
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]


            # print(f"Batch {batch_idx+1}/{num_batches}")
            # print(f"batch_input_ids shape: {batch_input_ids.shape}")
            # print(f"batch_attention_mask shape: {batch_attention_mask.shape}")

            # Get model outputs for the batch
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            batch_log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

            # Iterate over each example in the batch
            for i in range(batch_input_ids.size(0)):
                input_ids_example = batch_input_ids[i]
                log_probs_example = batch_log_probs[i]
                tokenized_test = tokenizer.convert_ids_to_tokens(input_ids_example)

                last_token = None
                last_token_log_prob = 0.0
                for j, token_id in enumerate(input_ids_example):
                    if token_id == tokenizer.pad_token_id:
                        break  # Stop processing at the first PAD token

                    token = tokenized_test[j]
                    token_log_prob = log_probs_example[j, token_id].item()  # Adjusted for batched log_probs

                    # Check if the current token is a suffix token
                    if token.startswith("##"):
                        if last_token is not None:
                            # Combine the last token (suffix or non-suffix) and current suffix token
                            token_pair = (last_token, token[2:])  # Remove '##' from the suffix token
                            pair_likelihoods[token_pair] += (token_log_prob + last_token_log_prob)
                            pair_freqs[token_pair] += 1

                    # Always update the last token, regardless of whether it's a suffix or non-suffix
                    last_token = token
                    last_token_log_prob = token_log_prob
                    
    return pair_likelihoods, pair_freqs



def compute_token_log_likelihoods(test_data, model, tokenizer):
    token_likelihoods = defaultdict(float)
    token_freqs = defaultdict(int)

    # Tokenize the test data and return PyTorch tensors
    test_encodings = tokenizer(test_data, truncation=True, padding=True, add_special_tokens=True, max_length=256, return_tensors='pt')

    # Move encodings to the same device as the model
    input_ids = test_encodings['input_ids'].to(model.device)
    attention_mask = test_encodings['attention_mask'].to(model.device)

    with torch.no_grad():
        for i in range(input_ids.size(0)):
            outputs = model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

            tokenized_test = tokenizer.convert_ids_to_tokens(input_ids[i])

            for j, token_id in enumerate(input_ids[i]):
                if token_id == tokenizer.pad_token_id:
                    break  # Stop processing at the first PAD token

                token = tokenized_test[j]
                token_log_prob = log_probs[0, j, token_id]
                token_likelihoods[token] += token_log_prob.item()
                token_freqs[token] += 1

    return token_likelihoods, token_freqs

    

# def strawberry_mutate(ls, tokenizer, training_data_list, test_data_list, trie, num_to_return=1, num_to_alter=1, include_greedy_optimal=False):
#     model = get_model(tokenizer, training_data_list)

#     word_likelihoods, word_freqs = compute_pair_log_likelihoods(test_data_list, model, tokenizer)

#     sorted_filtered_words = filter_and_sort_words(word_likelihoods, word_freqs, 2, tokenizer)

#     # Pre-calculate words and weights
#     words_and_weights = [(pair[0][0] + pair[0][1], pair[1].item()) for pair in sorted_filtered_words[:200]]
#     words, weights = zip(*words_and_weights)
#     weights = [weight ** 2 for weight in weights]  # Making weights positive
#     weights = np.array(weights)
#     weights /= weights.sum()  # Normalize so that the sum is 1

#     modified_lists = []
#     greedy_used = False

#     for _ in range(num_to_return):
#         if include_greedy_optimal and not greedy_used:
#             # Select top num_to_alter words for greedy approach
#             selected_words = [word for word, _ in words_and_weights[:num_to_alter]]
#             greedy_used = True
#         else:
#             # Select words based on weighted random choice
#             selected_words = list(np.random.choice(words, size=num_to_alter, replace=False, p=weights))

#         # Create a copy of ls for modification
#         modified_ls = ls.copy()

#         # Add selected words
#         for word in selected_words:
#             if word not in modified_ls:
#                 modified_ls.append(word)
                
#         toks_to_remove = tokens_to_remove_trie_freq(random.sample(ls, k=len(ls) // 10), trie, len(modified_ls) - len(ls), not_to_touch=selected_words)
#         # Remove tokens
#         for tok in toks_to_remove:
#             if tok in modified_ls:
#                 modified_ls.remove(tok)

#         # Sanity check to ensure the length is consistent
#         if len(modified_ls) == len(ls): 
#             modified_lists.append(modified_ls)

#     return modified_lists



def strawberry_mutate(ls, perma_tokens, temp_tokenizer, training_data_list, training_data_corpus, test_data_list, trie, num_to_return=1, num_to_alter=1, include_greedy_optimal=False, experiment_name='exp_default', multiple_removals=True):
    # Create a copy of ls for modification
    pruned_ls = ls.copy()
    toks_to_remove = (tokens_to_remove_corpus_freq(num_to_alter * 2, training_data_corpus, temp_tokenizer, ls, 0.1))
    if multiple_removals:
        toks_to_remove.extend(tokens_to_remove_trie_freq(random.sample(ls, k=len(ls) // 10), trie, num_to_alter))
        toks_to_remove.extend(tokens_to_remove_similar(ls, num_to_alter, trie))

    random.shuffle(toks_to_remove)
    # Remove tokens
    for tok in toks_to_remove:
        if len(pruned_ls) == len(ls) - num_to_alter:
            break
        if tok in pruned_ls:
            pruned_ls.remove(tok)

    tokenizer = get_tokenizer(pruned_ls + perma_tokens, file_append=experiment_name)

    model = get_model(tokenizer, training_data_list)

    word_likelihoods, word_freqs = compute_pair_log_likelihoods(test_data_list, model, tokenizer)

    sorted_filtered_words = filter_and_sort_words(word_likelihoods, word_freqs, 2, tokenizer)

    # Pre-calculate words and weights
    words_and_weights = [(pair[0][0] + pair[0][1], pair[1].item()) for pair in sorted_filtered_words[:200]]
    words, weights = zip(*words_and_weights)
    weights = [weight ** 2 for weight in weights]  # Making weights positive
    weights = np.array(weights)
    weights /= weights.sum()  # Normalize so that the sum is 1

    modified_lists = []
    greedy_used = False

    for _ in range(num_to_return):
        modified_ls = pruned_ls.copy()
        if include_greedy_optimal and not greedy_used:
            # Select top num_to_alter words for greedy approach
            selected_words = [word for word, _ in words_and_weights[:num_to_alter]]
            greedy_used = True
        else:
            # Select words based on weighted random choice
            selected_words = list(np.random.choice(words, size=max(num_to_alter * 2, len(words)), replace=False, p=weights))
        # print(selected_words, toks_to_remove)
        # print(selected_words[:num_to_alter])
        # Add selected words
        for word in selected_words:
            if word not in modified_ls:
                modified_ls.append(word)
            if len(modified_ls) == len(ls):
                break

        # Sanity check to ensure the length is consistent
        if len(modified_ls) == len(ls): 
            modified_lists.append(modified_ls)

    return modified_lists



def merge_and_assess(test_data, train_data, tokenizer, current_tokens, n, m, experiment_name='default_name', use_freq=False):
    # Step 1: Get data on loss incurred by pairs of tokens
    model = get_model(tokenizer, train_data)
    pair_loss_dict, pair_freq_dict = compute_pair_log_likelihoods(test_data, model, tokenizer)

    # Step 2: Sort pairs by loss
    if use_freq:
        sorted_pairs = sorted(pair_freq_dict.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_pairs = sorted(pair_loss_dict.items(), key=lambda x: x[1], reverse=False)

    # Modified Step 2: Select top n pairs ensuring uniqueness of tokens and store their frequencies
    top_n_pairs = []
    top_n_pair_frequencies = {}  # Store frequencies of the top n pairs
    used_tokens = set()
    for pair, loss in sorted_pairs:
        if pair[0] not in used_tokens and pair[1] not in used_tokens:
            top_n_pairs.append((pair, loss))
            top_n_pair_frequencies[''.join(pair)] = pair_freq_dict[pair]  # Store frequency
            used_tokens.update(pair)
        if len(top_n_pairs) == n:
            break
    # Step 3: Merge each of the top n pairs to form new tokens
    merged_tokens = [''.join(pair[0]) for pair in top_n_pairs]

    # Step 4: Combine new tokens with the original set of tokens
    all_tokens = list(current_tokens) + merged_tokens
    temp_tokenizer = get_tokenizer(all_tokens, file_append=experiment_name)
    temp_model = get_model(temp_tokenizer, train_data)
    # Step 5: Get data on loss incurred by individual tokens
    token_loss_dict, token_freq_dict = compute_token_log_likelihoods(test_data, temp_model, temp_tokenizer)  # Placeholder for actual function call

    # Step 6: Compare new loss vs old loss and calculate improvement
    improvement_dict = {}
    for pair, _ in top_n_pairs:
        merged_token = ''.join(pair)
        original_loss = pair_loss_dict[pair]
        new_loss = token_loss_dict[merged_token]
        improvement = new_loss - original_loss
        improvement_dict[merged_token] = (improvement, (original_loss, new_loss))

    # Step 7: Sort by improvement and select top m tokens
    sorted_improvements = sorted(improvement_dict.items(), key=lambda x: x[1][0], reverse=True)

    # ... [Previous steps remain the same]

    # Step 7: Sort by improvement and filter out tokens with a frequency of 0
    filtered_improvements = [(token, improvement) for token, improvement in sorted_improvements if token_freq_dict[token] > 0]

    # Modified: Select top m tokens from the filtered list, include original pair frequencies, and ensure token freq >= pair freq
    top_m_tokens_info = [
        (token, improvement_dict[token][0], improvement_dict[token][1], token_freq_dict[token], top_n_pair_frequencies.get(token, 'N/A'))
        for token, _ in filtered_improvements
        if token_freq_dict[token] >= top_n_pair_frequencies.get(token, 0) and improvement_dict[token][0] > 0
    ]

    # Select top m tokens after filtering
    top_m_tokens_info = top_m_tokens_info[:m]
    # print(top_m_tokens_info)


    # Update return to reflect the filtered list
    return [token_info[0] for token_info in top_m_tokens_info]




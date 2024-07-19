from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import json
from collections import Counter
import math
from tokenizers import BertWordPieceTokenizer
import os
from collections import defaultdict
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

def get_tokenizer(tok_ls, file_append=''):
    tok_ls = tok_ls.copy()
    tok_ls.append('[UNK]')
    tok_ls.append('[PAD]')

    # Step 1: Create vocab and save it to vocab.json
    vocab = {str(token): i for i, token in enumerate(tok_ls)}
    with open('tokenizer_files/vocab.json', 'w') as f:
        json.dump(vocab, f)

    # Step 2: Create the tokenizer model
    tokenizer_model = models.WordPiece(vocab=vocab, unk_token="[UNK]")
    tokenizer = Tokenizer(model=tokenizer_model)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()

    # Save the tokenizer
    tokenizer.save('tokenizer_files/' + file_append + "custom_tokenizer.json")

    # Step 3: Load the tokenizer using transformers
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file='tokenizer_files/' + file_append + "custom_tokenizer.json")
    tokenizer.pad_token = '[PAD]'
    tokenizer.unk_token = '[UNK]'
    return tokenizer

# def calculate_log_ngram_likelihood_with_smoothing(tokenized_corpus, n, k=1):
#     ngram_counts = defaultdict(int)
#     token_counts = defaultdict(int)
#     total_tokens = 0

#     # Count the occurrences of n-grams and tokens in the tokenized corpus
#     for i in range(len(tokenized_corpus) - n + 1):
#         ngram = tuple(tokenized_corpus[i:i+n])
#         ngram_counts[ngram] += 1

#         # Count individual tokens for add-k smoothing denominator
#         for token in ngram:
#             token_counts[token] += 1
#             total_tokens += 1

#     # Calculate the log probabilities of n-grams with add-k smoothing
#     log_ngram_probabilities = {}
#     for ngram, count in ngram_counts.items():
#         log_ngram_probabilities[ngram] = math.log((count + k) / (token_counts[ngram[-1]] + k * len(token_counts)))

#     # Compute the log likelihood of the tokenized corpus using log probabilities
#     log_corpus_likelihood = 0.0
#     for ngram in ngram_counts:
#         log_corpus_likelihood += log_ngram_probabilities.get(ngram, math.log(k / (token_counts[ngram[-1]] + k * len(token_counts))))

#     return log_corpus_likelihood

# n >= 2 
def get_ngram_probabilities_from_train_corpus(tokenized_train_corpus, n, k=1):
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)

    # Count n-grams and (n-1)-grams
    for i in range(len(tokenized_train_corpus) - n + 1):
        ngram = tuple(tokenized_train_corpus[i:i+n])
        ngram_counts[ngram] += 1
        context = ngram[:-1]
        context_counts[context] += 1

    # Calculate log probabilities
    log_ngram_probabilities = {}
    V_prime = len(context_counts)  # Number of unique (n-1)-grams
    for ngram, count in ngram_counts.items():
        
        context = ngram[:-1]
        denominator = context_counts[context] + k * V_prime
        log_ngram_probabilities[ngram] = math.log((count + k) / denominator)

    return log_ngram_probabilities, V_prime


def compute_log_likelihood(tokenized_test_corpus, tokenized_train_corpus, n, k=1):
    log_ngram_probabilities, V_prime = get_ngram_probabilities_from_train_corpus(tokenized_train_corpus, n, k)
    
    log_likelihood = 0.0

    context_denominator_unseen = k * V_prime  # Assumes unseen (n-1)-gram contexts have a count of 0
    default_log_prob_unseen = math.log(k / context_denominator_unseen)

    # Loop through the test corpus to compute likelihood
    for i in range(len(tokenized_test_corpus) - n + 1):
        ngram = tuple(tokenized_test_corpus[i:i+n])
        
        # If ngram is present in our probabilities, use it; otherwise use a default value
        if ngram in log_ngram_probabilities:
            log_likelihood += log_ngram_probabilities[ngram]
        else:
            log_likelihood += default_log_prob_unseen

    return log_likelihood



def custom_sort_and_count(strings):
    def key_func(s):
        if len(s) == 1 or (len(s) == 3 and s.startswith("##")):
            return (0, s)
        return (1, s)

    sorted_strings = sorted(strings, key=key_func)

    count = sum(1 for s in sorted_strings if len(s) == 1 or (len(s) == 3 and s.startswith("##")))

    return sorted_strings, count

def wordpiece_perm_generator(wanted_back, corpus_file_name="text_fic.txt", num_tokens=3000, exclude_singles=False, batch_size=10):
    # initialize the word_piece solution

  # Save the corpus to a text file (the trainer expects a file)

  # Initialize a tokenizer
  wp_tokenizer = BertWordPieceTokenizer()

  # Train the tokenizer
  wp_tokenizer.train(
      files=corpus_file_name,
      vocab_size=num_tokens,
      min_frequency=2,
      special_tokens=[
          "[PAD]",
          "[UNK]"
      ]
  )

  wp_ls = []
  # Load vocab and write tokens to a file
  for token, token_id in wp_tokenizer.get_vocab().items():
      wp_ls.append(token)
  if '[PAD]' in wp_ls:
    wp_ls.remove('[PAD]')
  if '[UNK]' in wp_ls:
    wp_ls.remove('[UNK]')
  random.shuffle(wp_ls)


  if exclude_singles:
    wp_ls, matching_count = custom_sort_and_count(wp_ls)
    if num_tokens - 2 - matching_count <= wanted_back:
      print('POTENTIAL ERROR')
    left = wp_ls[matching_count:]
    batch = []
    for i in range(batch_size):
      random.shuffle(left)
      batch.append(left[:wanted_back])
    return batch


  return wp_ls[:wanted_back]

def sort_lists_by_floats(list_of_lists, list_of_floats, list_of_ints):
    sorted_data = sorted(zip(list_of_lists, list_of_floats, list_of_ints), key=lambda x: x[1], reverse=True)
    sorted_lists, sorted_floats, sorted_ints = zip(*sorted_data)
    return list(sorted_lists), list(sorted_floats), list(sorted_ints)


def show_score_plots(scores):
  # print(scores[-1])
  # print(sum(scores[-1][:20]) / float(len(scores[-1][:20])))
  if len(scores) > 1 and scores[-1][:10] == scores[-2][:10]:
    print('NO CHANGE FROM PREV')
  # Create a list of 20 lists, each containing floats
  # For demonstration purposes, I'm filling each sublist with random floats.
  data = scores

  # Create a new figure and axis for the plot
  fig, ax = plt.subplots()

  # Loop through each of the 20 lists
  for i, sublist in enumerate(data[1:]):
      # Extract the first 10 floats from each list
      first_30_values = sublist[:10]

      # Plot these 10 floats
      ax.scatter([i+1]*10, first_30_values)

  # Customize the plot
  ax.set_xlabel('List Index')
  ax.set_ylabel('Float Value')
  ax.set_title('First 10 Floats in Each of 20 Lists')

  plt.xticks(np.arange(1, 101))  # set x-ticks to be from 1 to 20
  plt.grid(True)  # Show grid lines for easier readability

  # Show the plot
  plt.show()
  # plt.pause(0.001)
def average_index_and_frequency(nums):
    # Dictionary to store the sum of indices and frequency of each integer
    index_sum = {}
    frequency = {}
    
    # Dictionary to store the final result
    result = {}
    
    # Iterate through the list to populate index_sum and frequency
    for i, num in enumerate(nums):
        if num in index_sum:
            index_sum[num] += i
            frequency[num] += 1
        else:
            index_sum[num] = i
            frequency[num] = 1
    
    # Calculate average index and frequency for each integer
    for num in index_sum:
        avg_index = index_sum[num] / frequency[num]
        result[num] = (avg_index, frequency[num])
        
    return result
def weighted_average(indices, values):
    if len(values) != len(indices):
        return "Error: Lists must be of the same length"
    
    total_sum = 0
    total_weight = 0
    
    for value, index in zip(values, indices):
        
        weight = 1 / (index + 1)
        total_sum += weight * value
        total_weight += weight
    
    return total_sum / total_weight

def create_index_weight_dict(integers, weights):
    if len(integers) != len(weights):
        return "Error: Lists must be of the same length"
    
    index_weight_dict = {}
    
    for i, (integer, weight) in enumerate(zip(integers, weights)):
        if integer not in index_weight_dict:
            index_weight_dict[integer] = ([], [])
        
        index_weight_dict[integer][0].append(i)
        index_weight_dict[integer][1].append(weight)
        
    return index_weight_dict

def adjust_mutation_weights(nums, weights):
    # Ensure both lists have the same length
    if len(nums) != len(weights):
        return "Both lists should contain the same number of elements."
    
    # Get indices sorted by their corresponding weights
    sorted_indices = sorted(range(len(weights)), key=lambda k: weights[k])

    # Get the max and min weight indices
    max_w_idx = sorted_indices[-1]
    min_w_idx = sorted_indices[0]

    # Check if the highest weight should be incremented
    if nums.count(1) != len(nums) - 1:
        nums[max_w_idx] += 1

        # Subtract one from the element with the lowest weight
        if nums[min_w_idx] > 1:
            nums[min_w_idx] -= 1
        else:
            # If the lowest weighted element cannot be decremented, 
            # decrement the next lowest and so on
            for idx in sorted_indices[1:]:
                if nums[idx] > 1:
                    nums[idx] -= 1
                    break
    return nums


def save_to_npy(list_of_lists, list_of_scores, file_name="current_toks_and_scores.npy"):
    # Convert list_of_lists to a NumPy array of objects (since it's a list of lists)
    np_list_of_lists = np.array(list_of_lists, dtype=object)
    
    # Convert list_of_scores to a NumPy array of floats
    np_list_of_scores = np.array(list_of_scores, dtype=float)
    
    # Create a dictionary to store both arrays
    data = {'list_of_toks': np_list_of_lists, 'list_of_scores': np_list_of_scores}
    
    # Save the dictionary as a .npy file
    np.save(file_name, data)
    # To load the data back into your program:
    # loaded_data = np.load('current_toks_and_scores.npy', allow_pickle=True).item()  

def upgrade_elites_and_downgrade_goats(save_goats, save_goats_scores, save_goats_counts, save_elite, save_scores, save_counts):
    if not save_elite:
        return save_goats, save_goats_scores, save_goats_counts, save_elite, save_scores, save_counts

    # Find the indices of elite members with counts >= 5
    elite_indices = [i for i, count in enumerate(save_counts) if count >= 5]


    for elite_index in elite_indices:
        elite_score = save_scores[elite_index]

        # Iterate through the goats and check if any have lower scores
        for goat_index, goat_score in enumerate(save_goats_scores):
            if elite_score > goat_score:
                # Upgrade the elite and downgrade the goat
                save_goats[goat_index], save_elite[elite_index] = save_elite[elite_index], save_goats[goat_index]
                save_goats_scores[goat_index], save_scores[elite_index] = save_scores[elite_index], save_goats_scores[goat_index]
                save_goats_counts[goat_index], save_counts[elite_index] = save_counts[elite_index], save_goats_counts[goat_index]

    return save_goats, save_goats_scores, save_goats_counts, save_elite, save_scores, save_counts













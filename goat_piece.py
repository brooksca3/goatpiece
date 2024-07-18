from iteration_tools import wordpiece_perm_generator, custom_sort_and_count, get_tokenizer
from strawberry_mutations import count_specific_token_frequencies, compute_pair_log_likelihoods, filter_and_sort_words, merge_and_assess
from model_playground import get_model
import string
import random
import time
import numpy as np

def add_with_freq(num_to_add_freq, tokenizer, corpus, current_tokens):
    # Tokenize the corpus
    tokens = tokenizer.tokenize(corpus)

    # Dictionary to hold concatenated token pairs and their frequencies
    pair_freq = {}

    # Helper function to check if a token is a special token or punctuation
    def is_special_or_punctuation(token):
        return token in string.punctuation or token.startswith('[') and token.endswith(']')

    # Iterate through tokens to count frequencies of adjacent pairs
    for i in range(len(tokens) - 1):
        first_token, second_token = tokens[i], tokens[i + 1]

        # Skip pairs if either token is a special token or punctuation
        if is_special_or_punctuation(first_token) or is_special_or_punctuation(second_token) or not second_token.startswith('##'):
            continue

        # Concatenate, removing '##' from the beginning of the second token if present
        concatenated_pair = first_token + second_token.lstrip('##')
        if concatenated_pair in current_tokens:
            continue

        # Update the frequency count of the pair
        if concatenated_pair in pair_freq:
            pair_freq[concatenated_pair] += 1
        else:
            pair_freq[concatenated_pair] = 1

    # Sort the pairs by frequency and select the top num_to_add_freq pairs
    sorted_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_pairs[:num_to_add_freq]]

def add_with_loss(current_tokens, training_data_list, test_data_list, num_to_add_loss, experiment_name):
    tokenizer = get_tokenizer(current_tokens, file_append=experiment_name)
    model = get_model(tokenizer, training_data_list)
    word_likelihoods, word_freqs = compute_pair_log_likelihoods(test_data_list, model, tokenizer)
    sorted_filtered_words = filter_and_sort_words(word_likelihoods, word_freqs, 2, tokenizer)
    words_and_weights = [(pair[0][0] + pair[0][1], pair[1]) for pair in sorted_filtered_words[:num_to_add_loss]]
    words, weights = zip(*words_and_weights)
    return list(words)

def add_with_loss_difference(test_data, training_data, tokenizer, all_tokens, num_to_consider, max_num_to_return, experiment_name):
    ## will make this a method for now in case we have intermediate adjustments later
    toks = merge_and_assess(test_data, training_data, tokenizer, all_tokens, num_to_consider, max_num_to_return, experiment_name)
    return toks

def select_tokens_to_remove(corpus, tokenizer, current_tokens, perma_tokens, num_to_delete, include_random=False, rand_prop=0.0, retired_tokens=[]):
    tokens_of_interest = list(set(current_tokens) - set(perma_tokens))
    frequency_dict = count_specific_token_frequencies(corpus, tokenizer, tokens_of_interest)
    sorted_tuples = sorted(frequency_dict.items(), key=lambda item: item[1], reverse=False)
    # Extracting the keys from the top 'num' tuples
    lowest_freq_tokens = [key for key, value in sorted_tuples[:num_to_delete]]

    # if we want to remove random stuff too...
    if include_random:
        lowest_freq_tokens = lowest_freq_tokens[:int(num_to_delete * (1 - rand_prop))]
        num_random_tokens = num_to_delete - len(lowest_freq_tokens)
        tokens_of_interest = list(set(tokens_of_interest) - set(lowest_freq_tokens))
        tokens_of_interest = list(set(tokens_of_interest) - set(retired_tokens))
        random.shuffle(tokens_of_interest)
        return lowest_freq_tokens + tokens_of_interest[:num_random_tokens]
    else:
        return lowest_freq_tokens

def run_goat_piece(train_corpus_file, train_lines, test_lines, num_tokens_desired, use_loss_every, num_to_delete, num_to_add_loss, num_to_add_freq, FREQ_TRAIN_NUM, DELETE_TRAIN_NUM, LOSS_TRAIN_NUM, LOSS_TEST_NUM, include_random=False, retire_after = 5, experiment_name='default_name'):
    ## we just need num_tokens_init to be bigger than the number of perma tokens
    num_tokens_init = 2 * len(set(' '.join(train_lines)))
    wp_ls = wordpiece_perm_generator(num_tokens_init, num_tokens=num_tokens_init + 2, corpus_file_name=train_corpus_file)
    wp_ref, matching_count = custom_sort_and_count(wp_ls)
    # perma_tokens are the single chars which we use to start the process
    perma_tokens = wp_ref[:matching_count]
    current_tokens = perma_tokens.copy()

    total_tokens = len(current_tokens)
    num_iteration = 0
    delete_dict = {}
    new_tokens = []
    retired_tokens = []
    orig_num_to_delete = num_to_delete
    while total_tokens < num_tokens_desired:
        print(f"iteration: {num_iteration}, current toks: {len(current_tokens)}, {len(set(current_tokens))}")
        if num_iteration > 0 and orig_num_to_delete >= len(new_tokens):
            num_to_delete = len(new_tokens) // 2
        else:
            num_to_delete = orig_num_to_delete
        random.shuffle(train_lines)
        random.shuffle(test_lines)
        train_corpus_freq = ' [UNK] '.join(train_lines[: FREQ_TRAIN_NUM])
        train_corpus_for_deletion = ' '.join(train_lines[: DELETE_TRAIN_NUM])

        if num_iteration > 0 and len(current_tokens) - num_to_delete - 1 > len(perma_tokens):
            #########
            start_time = time.time()
            temp_tokenizer = get_tokenizer(current_tokens, file_append=experiment_name)
            ## added new_tokens here to the toks_of_interest to protect tokens which have just been added
            tokens_to_remove = select_tokens_to_remove(train_corpus_for_deletion, temp_tokenizer, current_tokens, perma_tokens + new_tokens, num_to_delete, include_random=include_random, rand_prop=0.5, retired_tokens=retired_tokens)
            print(f"tokens removed: {tokens_to_remove}")
            for tok in tokens_to_remove:
                if tok in current_tokens:
                    current_tokens.remove(tok)
                    if tok in delete_dict:
                        delete_dict[tok] += 1
                        if delete_dict[tok] >= retire_after:
                            retired_tokens.append(tok)
                    else:
                        delete_dict[tok] = 1
            end_time = time.time()
            print(f"deletion time: {end_time - start_time}")
            #########

        if (num_iteration + 1) % use_loss_every == 0 and len(current_tokens) > 1000:
            start_time = time.time()
            # new_tokens = add_with_loss(current_tokens, train_lines[:LOSS_TRAIN_NUM], test_lines[:LOSS_TEST_NUM], num_to_add_loss, experiment_name)
            tokenizer = get_tokenizer(current_tokens, file_append=experiment_name)
            new_tokens = add_with_loss_difference(test_lines[:LOSS_TEST_NUM], train_lines[:LOSS_TRAIN_NUM], tokenizer, current_tokens, 3 * num_to_add_loss, num_to_add_loss, experiment_name)
            end_time = time.time()
            print(f"loss add time: {end_time - start_time}")
            print(f"tokens added with loss: {new_tokens}")
        else:
            start_time = time.time()
            tokenizer = get_tokenizer(current_tokens, file_append=experiment_name)
            new_tokens = add_with_freq(num_to_add_freq, tokenizer, train_corpus_freq, current_tokens)
            end_time = time.time()
            print(f"freq add time: {end_time - start_time}")
            print(f"tokens added with freq: {new_tokens}")
        current_tokens.extend(new_tokens)
        num_iteration += 1
        total_tokens = len(current_tokens)
    
    ## after termination, ensure the proper size of the tokenization
    if len(current_tokens) > num_tokens_desired:
        diff = len(current_tokens) - num_tokens_desired
        temp_tokenizer = get_tokenizer(current_tokens, file_append=experiment_name)
        random.shuffle(train_lines)
        random.shuffle(test_lines)
        train_corpus_for_deletion = ' '.join(train_lines[: DELETE_TRAIN_NUM])
        tokens_to_remove = select_tokens_to_remove(train_corpus_for_deletion, temp_tokenizer, current_tokens, perma_tokens, diff)
        for tok in tokens_to_remove:
            if tok in current_tokens:
                current_tokens.remove(tok)
    return current_tokens


from iteration_tools import wordpiece_perm_generator, custom_sort_and_count, get_tokenizer
import numpy as np
from model_playground import calculate_perplexity_with_gpt2_loss
import random

num_tokens_init = 20_000
corpus_file_name = 'data/difference_exp_train.txt'

wp_ls = wordpiece_perm_generator(num_tokens_init, num_tokens=num_tokens_init + 2, corpus_file_name=corpus_file_name)
wp_ref, matching_count = custom_sort_and_count(wp_ls)

tester_wp_ls = wordpiece_perm_generator(4 * num_tokens_init, num_tokens=4 * num_tokens_init + 2, corpus_file_name=corpus_file_name)
tester_wp_ref, tester_matching_count = custom_sort_and_count(tester_wp_ls)
tester_wp_ref = tester_wp_ref[matching_count:]
random.shuffle(tester_wp_ref)
# perma_tokens are the single chars we need to keep :)
perma_tokens = wp_ref[:matching_count]
# wp_ref are the non-perma tokens for the wordpiece tokenization of the given corpus
wp_ref = wp_ref[matching_count:]
# this is our baseline, the wordpiece tokenizer
wordpiece_tokenizer = get_tokenizer(wp_ref + perma_tokens, file_append='realwp')
tester_tokenizer = get_tokenizer(tester_wp_ref[:len(wp_ref)] + perma_tokens, file_append='testerwp')

with open('data/difference_exp_train.txt', 'r') as f:
    train = f.read()
    train = train.split('\n')
    f.close()
with open('data/difference_exp_held_out.txt', 'r') as f:
    test = f.read()
    test = test.split('\n')
    f.close()
print(len(train))
print(len(test))

wp_score = calculate_perplexity_with_gpt2_loss(wordpiece_tokenizer, train[:int(0.5 * len(train))], test[:int(0.5 * len(test))])
print(f"wordpiece score: {wp_score}")
tester_score = calculate_perplexity_with_gpt2_loss(tester_tokenizer, train[:int(0.5 * len(train))], test[:int(0.5 * len(test))])
print(f"tester score: {tester_score}")





from datasets import load_dataset
import numpy as np
from goat_piece import run_goat_piece
from iteration_tools import wordpiece_perm_generator, get_tokenizer
import unicodedata
import re
import os
import sys

print('no random - this is to compare')

def deaccent(text):
    # Decompose the string into base characters and combining characters
    decomposed = unicodedata.normalize('NFD', text)
    # Filter out the combining characters (diacritics)
    stripped = ''.join(ch for ch in decomposed if unicodedata.category(ch) != 'Mn')
    return stripped.lower()

def pad_non_alphanumeric(input_str):
    return re.sub(r'([^\w\s])', r' \1 ', input_str)

if len(sys.argv) == 1:
    print("Received no arguments. Setting default parameters")
    EXPERIMENT_NAME = 'goat2_1k_lfg_no_random_2freq'
    NUM_TOKENS_DESIRED = 10_000
    USE_LOSS_EVERY = 2
    NUM_TO_DELETE = 25
    NUM_TO_ADD_LOSS = 50
    NUM_TO_ADD_FREQ = 50
    FREQ_TRAIN_NUM = 10_000
    DELETE_TRAIN_NUM = 10_000
    LOSS_TRAIN_NUM = 40_000
    LOSS_TEST_NUM = 2_000
else:
    print(f"Received arguments: {sys.argv}")
    EXPERIMENT_NAME = sys.argv[1]  
    NUM_TOKENS_DESIRED = int(sys.argv[2])
    USE_LOSS_EVERY = int(sys.argv[3])
    NUM_TO_DELETE = int(sys.argv[4])
    NUM_TO_ADD_LOSS = int(sys.argv[5])
    NUM_TO_ADD_FREQ = int(sys.argv[6])
    FREQ_TRAIN_NUM = int(sys.argv[7]) 
    DELETE_TRAIN_NUM = int(sys.argv[8])
    LOSS_TRAIN_NUM = int(sys.argv[9])
    LOSS_TEST_NUM = int(sys.argv[10])

## some other parameters
NUM_EXAMPLES = 100_000
MAX_TOKS_PER_INPUT = 256
TRAIN_PROPORTION = 0.5

if os.path.exists('data/difference_exp_test.txt'):
    with open('data/difference_exp_train.txt', 'r') as f:
       train_lines = f.readlines()
    with open('data/difference_exp_test.txt', 'r') as f:
       test_lines = f.readlines()
    x = run_goat_piece(train_corpus_file='data/difference_exp_train.txt', train_lines=train_lines, test_lines=test_lines, num_tokens_desired=NUM_TOKENS_DESIRED, use_loss_every=USE_LOSS_EVERY, num_to_delete=NUM_TO_DELETE, num_to_add_loss=NUM_TO_ADD_LOSS, num_to_add_freq=NUM_TO_ADD_FREQ, FREQ_TRAIN_NUM=FREQ_TRAIN_NUM, DELETE_TRAIN_NUM=DELETE_TRAIN_NUM, LOSS_TRAIN_NUM=LOSS_TRAIN_NUM, LOSS_TEST_NUM=LOSS_TEST_NUM, include_random=False, retire_after=3, experiment_name=EXPERIMENT_NAME)
    print(x)
x = np.array(x)
np.save(EXPERIMENT_NAME + '_.npy', x)
# python goat_piece_setup.py testerlol2 10000 2 10 20 20 20000 20000 20000 10000

# ds = load_dataset('wikitext', 'wikitext-103-v1')
# train_data = ds["train"]
# data_lines = []
# shuffled_data = train_data['text'][:]
# random.shuffle(shuffled_data)

# print('dataset loaded')
# ### dataset set up
# wp_ls = wordpiece_perm_generator(NUM_TOKENS_DESIRED, num_tokens=NUM_TOKENS_DESIRED+2, corpus_file_name='text_fic1.txt')
# tokenizer = get_tokenizer(wp_ls, file_append=EXPERIMENT_NAME)

# for t in shuffled_data:
#   t = deaccent(t)
#   t = pad_non_alphanumeric(t)
#   num_toks = len(tokenizer.encode(t))
#   if num_toks > 50 and num_toks < MAX_TOKS_PER_INPUT - 100:
#     data_lines.append(t.strip())
#   if len(data_lines) > NUM_EXAMPLES:
#     break

# train_lines = data_lines[:int(TRAIN_PROPORTION * len(data_lines))]
# test_lines = data_lines[int(TRAIN_PROPORTION * len(data_lines)):]

# ## save the current experiment's train and test
# with open(EXPERIMENT_NAME + '_train.txt', 'w') as f:
#   f.write('\n'.join(train_lines))
# with open(EXPERIMENT_NAME + '_test.txt', 'w') as f:
#   f.write('\n'.join(test_lines))
# # ##
# with open(EXPERIMENT_NAME + '_train.txt', 'r') as f:
#     corpus = f.read().lower()
#     f.close()

# x = run_goat_piece(train_corpus_file=EXPERIMENT_NAME + '_train.txt', train_lines=train_lines, test_lines=test_lines, num_tokens_desired=NUM_TOKENS_DESIRED, use_loss_every=USE_LOSS_EVERY, num_to_delete=NUM_TO_DELETE, num_to_add_loss=NUM_TO_ADD_LOSS, num_to_add_freq=NUM_EXAMPLES, FREQ_TRAIN_NUM=FREQ_TRAIN_NUM, DELETE_TRAIN_NUM=DELETE_TRAIN_NUM, LOSS_TRAIN_NUM=LOSS_TRAIN_NUM, LOSS_TEST_NUM=LOSS_TEST_NUM, experiment_name='default_name')

import re
def sent_too_long(sent, max_toks, tokenizer):
  # split up a list until no more than 500 tokens anywhere
  words = sent.split(' ')
  # tokenize each word, concatenate until too many tokens!
  final_ls = []
  cur_str = ''
  cur_toks = 0
  for word in words:
    toks = tokenizer(word)['input_ids']
    if (len(toks) + cur_toks) <= max_toks:
      cur_str += (' ' + word)
      cur_toks += len(toks)
    else:
      final_ls.append(cur_str)
      cur_str = word
      cur_toks = len(toks)
  if cur_toks > 0:
    final_ls.append(cur_str)
    return final_ls



# returns a list of sentences which we can write to a file, separated by newlines
def sep_sents(text, ind, tokenizer, max_toks=500):
#   print('IND: ' + str(ind))

  text = re.sub(r'\·', '· [SEP] ', text)
  text = re.sub(r'\.', '. [SEP] ', text)
  text = re.sub(r'\;', '; [SEP] ', text)
  text = re.sub(r'\!', '! [SEP] ', text)


  sents = text.split('[SEP]')
  # tokenize each sentence, concatenate until too many tokens!
  final_ls = []
  cur_str = ''
  cur_toks = 0
  for sent in sents:
    toks = tokenizer(sent)['input_ids']
    if (len(toks) + cur_toks) <= max_toks:
      cur_str += (' ' + sent)
      cur_toks += len(toks)
    elif cur_toks == 0:
      final_ls += sent_too_long(sent, max_toks, tokenizer)
      cur_toks = 0
      cur_str = ''
    else:
      final_ls.append(cur_str)
      cur_str = sent
      cur_toks = len(toks)
  if cur_toks > 0:
    final_ls.append(cur_str)
#   if ' ' not in cur_str:
    # print('HERE: ' + str(ind))
  ret_ls = []
  for el in final_ls:
    if len(tokenizer(el)['input_ids']) > max_toks:
      ret_ls += sep_sents(el, -1 * ind)
    else:
      ret_ls.append(el)
  return ret_ls
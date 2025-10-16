import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_file(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def initialize_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        token_id = 256 + i
        vocab[token_id] = token.encode("utf-8")
    return vocab
    #######change this for the second task to pass
def split_around_special_tokens(special_tokens, text,drop_special=True):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]
#####change this for third task pass

def pretokenization(chunks, special_tokens):
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []
    for chunk in chunks:  
        if chunk in special_tokens:
            tokens.append(chunk)
        else:
            tokens.extend(m.group(0) for m in re.finditer(PAT, chunk))
    return tokens

def word2bytes(word):
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)

def count_word(chunk):
    word_cnt = defaultdict(int)
    for m in re.finditer(PAT, chunk):
        word = m.group(0)
        bytes_tuple = word2bytes(word)
        word_cnt[bytes_tuple] += 1
    return word_cnt

def combine_word_dicts(word_dicts):
    merged = defaultdict(int)
    for d in word_dicts:
        for k, v in d.items():
            merged[k] += v 
    return merged

def pair_count(word_cnt):
    pair_cnt = defaultdict(int)
    for word_bytes, cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            pair_cnt[pair] += cnt
    return pair_cnt

def find_max_pair(pair_cnt):
    return max(pair_cnt.items(), key=lambda x: (x[1], x[0]))[0]

def merge(word_bytes, max_pair):
    new_pair = max_pair[0] + max_pair[1]
    new_word_bytes = []
    i = 0
    while i < len(word_bytes):
        if i < len(word_bytes) - 1 and word_bytes[i] == max_pair[0] and word_bytes[i+1] == max_pair[1]:
            new_word_bytes.append(new_pair) 
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)

def update(word_cnt, pair_cnt, max_pair):
    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt)
    for word_bytes, cnt in word_cnt.items():
        ### update word_cnt
        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
        if max_pair not in old_pairs:
            new_word_cnt[word_bytes] += cnt
            continue
        new_word = merge(word_bytes, max_pair)
        new_word_cnt[new_word] += cnt 
        ### update pair_cnt
        for pair in old_pairs:
            new_pair_cnt[pair] -= cnt 
            if new_pair_cnt[pair] == 0:
                del new_pair_cnt[pair]
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt
    return new_word_cnt, new_pair_cnt

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = read_file(input_path)
    vocab = initialize_vocab(special_tokens)
    chunks = split_around_special_tokens(special_tokens, text)
    if len(chunks) < 4: 
        word_dicts = list(map(count_word, chunks))
    else: 
        word_dicts = process_map(count_word, chunks, chunksize=1)
    word_cnt = combine_word_dicts(word_dicts)
    pair_cnt = pair_count(word_cnt)
    merges = []
    while len(vocab) < vocab_size:
        max_pair = find_max_pair(pair_cnt)
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        word_cnt, pair_cnt = update(word_cnt, pair_cnt, max_pair)
    return vocab, merges



from typing import Iterator, Iterable
import json

def split_words(text):
    raw_tokens = []
    raw_tokens.extend(m.group(0) for m in re.finditer(PAT, text))
    return raw_tokens

def apply_merges(word_bytes, merges_set, vocab_to_id):
    word_bytes = list(word_bytes)
    
    while True:
        min_token_id = float('inf')
        best_pair_idx = -1
        merged = None

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined

        if best_pair_idx == -1:
            break

        # Apply best merge
        word_bytes = (
            word_bytes[:best_pair_idx]
            + [merged]
            + word_bytes[best_pair_idx + 2:]
        )

    return tuple(word_bytes)

def encode_merged(text, merges, reverse_vocab):
    word_list = split_words(text)
    tokens = []
    for word in word_list:
        word_bytes = list(word2bytes(word))
        merged_word_bytes = apply_merges(word_bytes, merges, reverse_vocab)
        tokens.extend(reverse_vocab[i] for i in merged_word_bytes)
    return tokens
class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode('utf-8') for i in self.special_tokens]
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.reverse_vocab = {v:k for k, v in vocab.items()}
        for special_token_byte in self.special_tokens_bytes:
            if special_token_byte not in self.reverse_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = special_token_byte
                self.reverse_vocab[special_token_byte] = new_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab_data = json.load(vf)
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v) for k, v in vocab_data.items()}
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            lines = mf.readlines()
            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        chunks = split_around_special_tokens(self.special_tokens, text, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.reverse_vocab[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.reverse_vocab))
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
    
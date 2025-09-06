# 训练BPE、序列化vocab/merges
import os
import regex as re
from typing import Iterable, List, Tuple, Dict
from collections import defaultdict, Counter
import json
from tqdm import tqdm
from multiprocessing import Pool, get_context

def init_worker(pat: str):
    global PAT_COMPILED
    PAT_COMPILED = re.compile(pat)

Pair = Tuple[bytes, bytes]  # (A, B)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = None
text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest <|endoftext|>"

    
def split_on_special(text: str, special_tokens: List[str]) -> List[str]:
    """按特殊标记切分，保证不跨特殊标记边界做 BPE 合并。"""
    if not special_tokens:
        return [text]
    # 逐个特殊标记转义再拼接为 alternation
    alt = "|".join(re.escape(tok) for tok in special_tokens)
    # 保留分隔符：用捕获括号，split 后分隔符也在结果里
    parts = re.split(f"({alt})", text)
    return [p for p in parts if p != ""]


def pre_tokenize_iter(path: str, special_tokens: List[str]) -> Iterable[str]:
    """
    流式产出预分词字符串（未转 bytes），不落地存储，节省内存。
    Also, special tokens are not yielded.
    """
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Contractions（英语缩略）：匹配 's, 't, 'd, 'm, 'll, 've, 're
    # 匹配词（letter sequences），并把紧邻其左侧的单个空格并入这个 token（GPT-2 的"空格黏右侧"策略）。 " hello"
    
    pat = re.compile(pat)
    
    # 获取文件大小用于进度监控
    file_size = os.path.getsize(path)
    
    with open(path, 'rb') as f:
        # 使用 tqdm 进度条
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="预分词处理")
        
        for line_num, line_bytes in enumerate(f, 1):
            # 更新进度条
            pbar.update(len(line_bytes))
            
            # 解码行内容
            try:
                line = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                continue  # 跳过无法解码的行
            line = split_on_special(line, special_tokens)
            for segment in line:
                if segment not in special_tokens:
                    yield from re.finditer(pat, segment)
        
        pbar.close()
            
def word_to_bytes(word: str) -> Tuple[bytes, ...]:
    # print(bytes(word.encode('utf-8')))
    enc = word.encode('utf-8')
    return tuple(enc[i:i+1] for i in range(len(enc)))

def to_tuple_dict(word_freqs: Dict[str, int]) -> Dict[Tuple[bytes, ...], int]:
    out = {}
    for w, f in word_freqs.items():
        out[word_to_bytes(w)] = f
    return out

def count_adjacent_pairs(tokenized_counts: Dict[Tuple[bytes, ...], int]) -> Counter[Pair]:
    pair_freq = Counter()
    for seq, f in tokenized_counts.items():
        for i in range(len(seq) - 1):
            pair_freq[(seq[i], seq[i+1])] += f
    return pair_freq

def select_best_pair(pair_freq: Counter[Pair]) -> Pair | None:
    if not pair_freq:
        return None
    # max 先比 count，再比 pair 的字典序（满足“并列时取字典序更大”）
    return max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))[0]

def merge_once_on_counts(tokenized_counts: Dict[Tuple[bytes, ...], int],
                         pair: Pair) -> Dict[Tuple[bytes, ...], int]:
    A, B = pair
    new_counts: Dict[Tuple[bytes, ...], int] = {}
    for seq, f in tokenized_counts.items():
        i = 0
        out: List[bytes] = []
        L = len(seq)
        while i < L:
            if i < L - 1 and seq[i] == A and seq[i+1] == B:
                out.append(A + B)  # 合并为一个 bytes 段，成为“完整 token”
                i += 2
            else:
                out.append(seq[i])
                i += 1
        tup = tuple(out)
        new_counts[tup] = new_counts.get(tup, 0) + f
    return new_counts

def save_vocab_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], output_path: str):
    os.makedirs(output_path, exist_ok=True)
    merges_path = output_path + "merges.txt"
    vocab_path = output_path + "vocab.json"
    # with open(merges_path, 'w') as f:
    #     for i, merge in enumerate(merges):
    #         f.write(f"{merge[0]} {merge[1]}\n")
    with open(merges_path, 'w') as f:
        merges_data = {
            "merges": [[int(b) for b in merge] for merge in merges]
        }
        json.dump(merges_data, f)
    with open(vocab_path, 'w') as f:
        vocab_serializable = {str(k): v.hex() for k, v in vocab.items()}
        json.dump(vocab_serializable, f)
        # vocab_serializable = {str(k): v.decode('utf-8', errors='replace') for k, v in vocab.items()}
        # json.dump(vocab_serializable, f)


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    merged : List[Tuple[bytes, bytes]] = []
    vocab_map : Dict[Tuple[bytes, bytes], int] = {}
    if special_tokens is None:
        special_tokens = []
    # 1. 初始化词汇表 - 256个字节值
    vocab = {i: bytes([i]) for i in range(256)}
    # 2. 添加特殊tokens到词汇表
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    # 3. initialize pre_tokenized word
    word_freqs = defaultdict(int)
    for token in pre_tokenize_iter(input_path, special_tokens):
        word_freqs[word_to_bytes(token.group())] += 1
        # if len(word_freqs) == 6:
            # break
        
    merge_times = vocab_size - len(vocab)
    for i in tqdm(range(merge_times), desc="训练BPE"):
        pair = count_adjacent_pairs(word_freqs)
        best_pair = select_best_pair(pair)
        if best_pair is None:
            print("没有更多的配对可以合并")
            break
        merged.append(best_pair)
        vocab_map[best_pair[0] + best_pair[1]] = len(vocab)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        new_word_freqs = merge_once_on_counts(word_freqs, best_pair)
        word_freqs = new_word_freqs
    print("合并列表", merged)
    print("词汇映射", vocab_map)
    print("词汇表", vocab)
    return vocab, merged
                   
                        
                        
if __name__ == "__main__":
    # trainer = BPE_Trainer(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=500, special_tokens=["<|endoftext|>"])
    # split_on_special("low low low low low lower lower widest widest widest newest newest newest newest newest newest <|endoftext|> nice to meet you", ["<|endoftext|>"])
    # i = iter(pre_tokenize_iter("data/TinyStoriesV2-GPT4-valid.txt", ["<|endoftext|>"]))
    # for token in i:
    #     print(token)
    # vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    vocab, merges = train_bpe("data/debug.txt", 10000, ["<|endoftext|>"])
    # save_vocab_merges(vocab, merges, "data/TinyStories/")
    
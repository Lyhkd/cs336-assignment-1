# 训练BPE、序列化vocab/merges
import os
import regex as re
from typing import Iterable, List, Tuple, Dict, BinaryIO
from collections import defaultdict, Counter
import json
from tqdm import tqdm
from multiprocessing import Pool, get_context
import multiprocessing

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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件分割成可以独立处理的块。
    可能返回较少的块数如果边界重叠。
    """
    assert isinstance(split_special_token, bytes), "特殊标记必须是字节字符串"

    # 获取文件总大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 初始块边界位置猜测，均匀分布
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次读取4k字节

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从边界猜测位置开始
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个小块

            # 如果到达文件末尾，边界应该在文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊标记
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 确保所有边界都是唯一的，但可能少于desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk_parallel(args):
    """
    并行处理单个文本块的预分词化
    """
    chunk_text, special_tokens, pat_str = args
    pat = re.compile(pat_str)
    word_freqs = defaultdict(int)
    
    # 按特殊标记分割
    segments = split_on_special(chunk_text, special_tokens)
    
    for segment in segments:
        if segment not in special_tokens:
            # 对每个段进行预分词
            for match in re.finditer(pat, segment):
                word_freqs[word_to_bytes(match.group())] += 1
    
    return dict(word_freqs)


def pre_tokenize_iter(path: str, special_tokens: List[str]) -> Iterable[str]:
    """
    流式产出预分词字符串（未转 bytes），不落地存储，节省内存。
    Also, special tokens are not yielded.
    """
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Contractions（英语缩略）：匹配 's, 't, 'd, 'm, 'll, 've, 're
    # 匹配词（letter sequences），并把紧邻其左侧的单个空格并入这个 token（GPT-2 的"空格黏右侧"策略）。 " hello"
    
    pat = re.compile(pat)
    
    # 修复文件读取问题：正确处理换行符
    with open(path, 'r', encoding='utf-8', newline='') as f:
        # 使用 tqdm 进度条
        file_size = os.path.getsize(path)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="预分词处理")
        
        for line_num, line in enumerate(f, 1):
            # 更新进度条
            pbar.update(len(line.encode('utf-8')))
            
            # 按特殊标记分割
            segments = split_on_special(line, special_tokens)
            for segment in segments:
                if segment not in special_tokens:
                    yield from re.finditer(pat, segment)
        
        pbar.close()


def pre_tokenize_parallel(path: str, special_tokens: List[str], num_processes: int = None) -> Dict[Tuple[bytes, ...], int]:
    """
    并行预分词化：使用多进程处理大文件以提高性能
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 获取文件分块边界
    with open(path, 'rb') as f:
        split_token = "<|endoftext|>".encode('utf-8') if "<|endoftext|>" in special_tokens else b"\n"
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
        
        # 准备并行处理的参数
        chunks_args = []
        
        print(f"将文件分割成 {len(boundaries)-1} 个块进行并行处理...")
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            try:
                chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
                chunks_args.append((chunk_text, special_tokens, pat_str))
            except UnicodeDecodeError:
                continue
    
    # 并行处理
    print(f"使用 {num_processes} 个进程并行处理...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_chunk_parallel, chunks_args),
            total=len(chunks_args),
            desc="并行预分词"
        ))
    
    # 合并结果
    print("合并处理结果...")
    combined_word_freqs = defaultdict(int)
    for chunk_result in results:
        for word_bytes, freq in chunk_result.items():
            combined_word_freqs[word_bytes] += freq
    
    return dict(combined_word_freqs)
            
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


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = None, use_parallel: bool = True, num_processes: int = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    merged : List[Tuple[bytes, bytes]] = []
    vocab_map : Dict[Tuple[bytes, bytes], int] = {}
    if special_tokens is None:
        special_tokens = []
    
    # 1. 初始化词汇表 - 256个字节值
    vocab = {i: bytes([i]) for i in range(256)}
    
    # 2. 添加特殊tokens到词汇表
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    
    # 3. 选择预分词化方法：并行或串行
    if use_parallel and os.path.getsize(input_path) > 1024 * 1024:  # 如果文件大于1MB则使用并行处理
        print("使用并行预分词化...")
        word_freqs = pre_tokenize_parallel(input_path, special_tokens, num_processes)
    else:
        print("使用串行预分词化...")
        word_freqs = defaultdict(int)
        for token in pre_tokenize_iter(input_path, special_tokens):
            word_freqs[word_to_bytes(token.group())] += 1
        word_freqs = dict(word_freqs)
    
    print(f"预分词化完成，共得到 {len(word_freqs)} 个不同的token")
    
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
    
    return vocab, merged
                   
                        
                        
if __name__ == "__main__":
    import time
    
    # 测试数据文件
    test_file = "test_newlines.txt"
    
    print("="*50)
    print("正在测试优化后的BPE训练...")
    print("="*50)
    
    # 测试并行处理
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=test_file, 
        vocab_size=1000, 
        special_tokens=["<|endoftext|>"],
        use_parallel=True,
        num_processes=4
    )
    parallel_time = time.time() - start_time

    
    # 保存结果

    
import json
from typing import Iterable, List, Dict, Tuple, Set
import regex as re
import os
from tqdm import tqdm

class UltraFastTokenizer:
    """超高性能Tokenizer实现"""
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str]):
        self.vocab = vocab
        self.special_tokens = set(special_tokens) if special_tokens else set()
        
        # 构建反向词汇表 - 使用更快的查找
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        
        # 预编译正则表达式
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 构建高效的merge规则
        self._build_efficient_merges(merges)
        
        # 构建特殊token的快速匹配
        self._build_special_token_matcher()
        
        # 预计算单字节映射
        self.byte_to_id = {}
        for i in range(256):
            byte_token = bytes([i])
            if byte_token in self.inverse_vocab:
                self.byte_to_id[i] = self.inverse_vocab[byte_token]
            else:
                self.byte_to_id[i] = i  # 回退到字节值
    
    def _build_efficient_merges(self, merges: List[Tuple[bytes, bytes]]):
        """构建高效的merge查找结构"""
        # 使用字典而不是列表，便于O(1)查找
        self.merge_ranks = {}
        self.merges = merges
        
        for rank, (first, second) in enumerate(merges):
            # 使用元组作为key，便于快速查找
            pair = (first, second)
            self.merge_ranks[pair] = rank
    
    def _build_special_token_matcher(self):
        """构建特殊token的高效匹配器"""
        if not self.special_tokens:
            self.special_regex = None
            return
        
        # 按长度排序，确保最长匹配
        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        pattern = '(' + '|'.join(escaped_tokens) + ')'
        self.special_regex = re.compile(pattern)
    
    def _split_with_special_tokens(self, text: str) -> List[str]:
        """使用特殊token分割文本"""
        if not self.special_regex:
            return [text]
        
        parts = []
        last_end = 0
        
        for match in self.special_regex.finditer(text):
            start, end = match.span()
            
            # 添加特殊token之前的文本
            if start > last_end:
                parts.append(text[last_end:start])
            
            # 添加特殊token
            parts.append(match.group())
            last_end = end
        
        # 添加最后剩余的文本
        if last_end < len(text):
            parts.append(text[last_end:])
        
        return [part for part in parts if part]
    
    def _get_pairs(self, word_bytes: List[bytes]) -> Set[Tuple[bytes, bytes]]:
        """获取所有相邻字节对"""
        pairs = set()
        for i in range(len(word_bytes) - 1):
            pairs.add((word_bytes[i], word_bytes[i + 1]))
        return pairs
    
    def _bpe_encode_optimized(self, text_bytes: bytes) -> List[int]:
        """优化的BPE编码算法"""
        if len(text_bytes) == 0:
            return []
        
        if len(text_bytes) == 1:
            return [self.byte_to_id[text_bytes[0]]]
        
        # 初始化为单字节列表
        word = [text_bytes[i:i+1] for i in range(len(text_bytes))]
        
        if len(word) == 1:
            token = word[0]
            return [self.inverse_vocab.get(token, self.byte_to_id[token[0]])]
        
        # 迭代应用merge规则
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # 找到优先级最高的pair
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # 应用merge
            first, second = best_pair
            new_word = []
            i = 0
            
            while i < len(word):
                if (i < len(word) - 1 and 
                    word[i] == first and 
                    word[i + 1] == second):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        
        # 转换为token IDs
        ids = []
        for token in word:
            if token in self.inverse_vocab:
                ids.append(self.inverse_vocab[token])
            else:
                # 回退到字节级别
                for byte_val in token:
                    ids.append(self.byte_to_id[byte_val])
        
        return ids
    
    def encode(self, text: str) -> List[int]:
        """主编码函数"""
        if not text:
            return []
        
        all_ids = []
        
        # 处理特殊token
        text_parts = self._split_with_special_tokens(text)
        
        for part in text_parts:
            if part in self.special_tokens:
                # 编码特殊token
                special_bytes = part.encode('utf-8')
                if special_bytes in self.inverse_vocab:
                    all_ids.append(self.inverse_vocab[special_bytes])
                continue
            
            # 预分词
            for match in self.pattern.finditer(part):
                token = match.group()
                token_bytes = token.encode('utf-8')
                
                # BPE编码
                token_ids = self._bpe_encode_optimized(token_bytes)
                all_ids.extend(token_ids)
        
        return all_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """批量编码，减少函数调用开销"""
        return [self.encode(text) for text in texts]
    
    def decode(self, ids: List[int]) -> str:
        """解码函数"""
        try:
            byte_sequence = b''.join(self.vocab[token_id] for token_id in ids)
            return byte_sequence.decode('utf-8', errors='replace')
        except KeyError:
            # 处理未知token ID
            byte_sequence = b''
            for token_id in ids:
                if token_id in self.vocab:
                    byte_sequence += self.vocab[token_id]
                else:
                    # 回退处理
                    byte_sequence += bytes([token_id % 256])
            return byte_sequence.decode('utf-8', errors='replace')
    
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: List[str]):
        """从文件加载tokenizer"""
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = {}
        for token_id_str, token_str in vocab_data.items():
            token_id = int(token_id_str)
            token_bytes = token_str.encode('utf-8')
            vocab[token_id] = token_bytes
        
        # 加载merge规则
        merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析merge规则，假设格式为 b'...' b'...'
                    # 使用更robust的解析方法
                    if line.startswith("b'") or line.startswith('b"'):
                        # 找到两个b'...'部分
                        parts = []
                        i = 0
                        while i < len(line):
                            if line[i:i+2] in ["b'", 'b"']:
                                quote = line[i+1]
                                start = i + 2
                                end = start
                                
                                # 找到对应的结束引号
                                while end < len(line) and line[end] != quote:
                                    if line[end] == '\\' and end + 1 < len(line):
                                        end += 2  # 跳过转义字符
                                    else:
                                        end += 1
                                
                                if end < len(line):
                                    # 解码字符串
                                    raw_str = line[start:end]
                                    # 处理转义字符
                                    decoded_str = raw_str.encode('utf-8').decode('unicode_escape')
                                    parts.append(decoded_str.encode('utf-8'))
                                    i = end + 1
                                else:
                                    break
                            else:
                                i += 1
                        
                        if len(parts) == 2:
                            merges.append((parts[0], parts[1]))
                    
                except Exception as e:
                    print(f"Warning: Failed to parse merge rule at line {line_num}: {line}")
                    print(f"Error: {e}")
                    continue
        
        print(f"Loaded {len(vocab)} vocab items and {len(merges)} merge rules")
        return cls(vocab, merges, special_tokens)
    
if __name__ == "__main__":
    vocab_path = "/data/yuer/assignment1-basics/data/TinyStories_train/vocab_str.json"
    merges_path = "/data/yuer/assignment1-basics/data/TinyStories_train/merges_str.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = UltraFastTokenizer.from_files(vocab_path, merges_path, special_tokens)
    with open("data/debug.txt", "r") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
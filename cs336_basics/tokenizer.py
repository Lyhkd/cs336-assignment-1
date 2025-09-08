import json
from typing import Iterable, List
import regex as re
import os
from tqdm import tqdm


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or {}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.inverse_vocab = self._id_to_vocab_bytes()
    
    
    def _id_to_vocab_bytes(self):
        return {v: k for k, v in self.vocab.items()}
    
    # def split_on_special(self, text: str, special_tokens: List[str]) -> List[str]:
    #     """按特殊标记切分，保证不跨特殊标记边界做 BPE 合并。"""
    #     if not special_tokens:
    #         return [text]
    #     # 逐个特殊标记转义再拼接为 alternation
    #     alt = "|".join(re.escape(tok) for tok in special_tokens)
    #     # 保留分隔符：用捕获括号，split 后分隔符也在结果里
    #     parts = re.split(f"({alt})", text)
    #     return [p for p in parts if p != ""]
    def split_on_special(self, text: str, special_tokens: List[str]) -> List[str]:
        """按特殊标记切分，保证不跨特殊标记边界做 BPE 合并。使用最长匹配优先。"""
        if not special_tokens:
            return [text]
        
        # 按长度降序排序，确保最长匹配优先
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        
        result = []
        i = 0
        while i < len(text):
            matched = False
            # 尝试匹配每个特殊标记（已按长度排序）
            for token in sorted_tokens:
                if text[i:i+len(token)] == token:
                    result.append(token)
                    i += len(token)
                    matched = True
                    break
            
            if not matched:
                # 找到下一个特殊标记的位置
                next_special_pos = len(text)
                for token in sorted_tokens:
                    pos = text.find(token, i)
                    if pos != -1 and pos < next_special_pos:
                        next_special_pos = pos
                
                # 添加普通文本
                if next_special_pos > i:
                    result.append(text[i:next_special_pos])
                    i = next_special_pos
                else:
                    # 没有找到更多特殊标记，添加剩余文本
                    result.append(text[i:])
                    break
        
        return [p for p in result if p != ""]

    def pre_tokenize_iter(self, line: str, special_tokens: List[str]) -> Iterable[str]:
        pat = re.compile(self.PAT)
        # 修复文件读取问题：正确处理换行符
        yield from re.finditer(pat, line)
                
    @staticmethod
    def from_files(vocab_path: str, merges_path: str, special_tokens: list[str]):
        with open(vocab_path, 'r') as f:
            vocab_str = json.load(f)
        vocab = {}
        for k, v in vocab_str.items():
            vocab[int(k)] = v.encode('utf-8')
        merges = []
        with open(merges_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 使用正则表达式匹配 b'...' b'...' 或 b"..." b"..." 格式
                    # 处理引号内可能包含各种字符的情况
                    pattern = r"b(['\"])(.*?)\1\s+b(['\"])(.*?)\3"
                    match = re.match(pattern, line)
                    if match:
                        # 提取两个部分并转换为bytes
                        first_part = match.group(2)
                        second_part = match.group(4)
                        a = first_part.encode('utf-8')
                        b = second_part.encode('utf-8')
                        merges.append((a, b))
        return Tokenizer(vocab, merges, special_tokens)
    
    
    def _encode_one_pretoken(self, token: str) -> List[int]:
        b = token.encode("utf-8")
        seq = [b[i:i+1] for i in range(len(b))]  # list[bytes], 每个元素是单字节
        for a, bb in self.merges:
            if not seq or len(seq) == 1:
                break
            seq = self._apply_one_merge(seq, a, bb)

        # 映射到 id，健壮性：若片段不在词表，退回逐字节
        ids: List[int] = []
        for chunk in seq:
            # print(chunk)
            tid = self.inverse_vocab.get(chunk)
            if tid is None:
                # 回退到单字节
                for byte in chunk:
                    tid_b = self.inverse_vocab.get(bytes([byte]))
                    if tid_b is None:
                        tid_b = int(byte)
                        # 理论不应发生（词表应含 256 单字节）；兜底
                        # raise KeyError(f"Byte {byte} missing in vocab.")
                    ids.append(tid_b)
            else:
                ids.append(tid)
        return ids
    
    @staticmethod
    def _apply_one_merge(seq: List[bytes], a: bytes, b: bytes) -> List[bytes]:
        out = []
        i = 0
        ab = a + b
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i+1] == b:
                out.append(ab)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out
    
    def encode(self, text: str) -> list[int]:
        ids: List[int] = []
        for segment in self.split_on_special(text, self.special_tokens):
            if segment in self.special_tokens:
                ids.append(self.inverse_vocab[segment.encode("utf-8")])
            else:
                for word in self.pre_tokenize_iter(segment, self.special_tokens):
                    ids.extend(self._encode_one_pretoken(word.group()))
        return ids
    
    def decode(self, ids: list[int]) -> str:
        byte_seq = b''.join(self.vocab[id] for id in ids)
        return byte_seq.decode('utf-8', errors='replace')  # errors 参数可以避免报错
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    
    
if __name__ == "__main__":
    vocab_path = "/data/yuer/assignment1-basics/data/TinyStories_train/vocab_str.json"
    merges_path = "/data/yuer/assignment1-basics/data/TinyStories_train/merges_str.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    with open("data/debug.txt", "r") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
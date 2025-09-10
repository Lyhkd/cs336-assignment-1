import json
import random
import os
from cs336_basics.optimized_tokenizer import Tokenizer, UltraFastTokenizer
import time


def benchmark_encode(tokenizer, docs, label: str, warmup: int = 1):
    # 预热，避免首次调用带来的抖动
    for _ in range(warmup):
        for d in docs:
            tokenizer.encode(d)

    start = time.perf_counter()
    total_bytes = 0
    total_tokens = 0
    for d in docs:
        total_bytes += len(d.encode('utf-8'))
        total_tokens += len(tokenizer.encode(d))
    elapsed = time.perf_counter() - start
    
    # 计算读取速度
    bytes_per_sec = total_bytes / elapsed if elapsed > 0 else float('inf')
    kb_per_sec = bytes_per_sec / 1024
    mb_per_sec = kb_per_sec / 1024

    print(f"\n=== {label} 读取速度基准 ===")
    print(f"总字节数: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"总 tokens: {total_tokens:,}")
    print(f"总耗时: {elapsed:.3f}s")
    print(f"读取速度: {bytes_per_sec:.0f} bytes/s ({kb_per_sec:.1f} KB/s, {mb_per_sec:.2f} MB/s)")

def sample_documents(file_path, num_samples=10):
    """从大文件中随机采样指定数量的文档（流式处理，避免内存爆掉）"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            print(f"处理第 {i} 行...")
            line = line.strip()
            if not line:  # 跳过空行
                continue
            if len(samples) < num_samples:
                samples.append(line)
            else:
                # 以 1/i 的概率替换已有样本
                j = random.randint(0, i - 1)
                if j < num_samples:
                    samples[j] = line
    return samples

def head_k(file_path, k=10):
    out = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
                if len(out) >= k:
                    break
    return out

def calculate_compression_ratio(text, token_ids):
    """计算压缩比：字节数/令牌数"""
    original_bytes = len(text.encode('utf-8'))
    num_tokens = len(token_ids)
    
    if num_tokens == 0:
        return float('inf')  # 避免除零错误
    
    return original_bytes / num_tokens

def main():
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 文件路径
    tinystories_path = "/data/yuer/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    owt_path = "/data/yuer/assignment1-basics/data/owt_train.txt"
    
    # Tokenizer路径
    ts_vocab_path = "/data/yuer/assignment1-basics/data/TinyStories_train/vocab_str.json"
    ts_merges_path = "/data/yuer/assignment1-basics/data/TinyStories_train/merges_str.json"
    
    owt_vocab_path = "/data/yuer/assignment1-basics/data/OWT_train/vocab_str.json"
    owt_merges_path = "/data/yuer/assignment1-basics/data/OWT_train/merges_str.json"
    
    special_tokens = ["<|endoftext|>"]
    
    print("正在加载tokenizer...")
    
    # 加载TinyStories tokenizer (10K词汇)
    ts_tokenizer = UltraFastTokenizer.from_files(ts_vocab_path, ts_merges_path, special_tokens)
    
    # 加载OpenWebText tokenizer (32K词汇)
    owt_tokenizer = UltraFastTokenizer.from_files(owt_vocab_path, owt_merges_path, special_tokens)
    
    print("正在采样文档...")
    
    # 采样文档
    ts_docs = head_k(tinystories_path, 100)
    owt_docs = head_k(owt_path, 100)
    
    # print(f"采样了 {len(ts_docs)} 个TinyStories文档")
    # print(f"采样了 {len(owt_docs)} 个OpenWebText文档")
    
    # # 分析TinyStories文档
    # print("\n=== TinyStories文档分析 (使用10K词汇tokenizer) ===")
    # ts_compression_ratios = []
    
    # for i, doc in enumerate(ts_docs):
    #     # 使用TinyStories tokenizer编码
    #     token_ids = ts_tokenizer.encode(doc)
    #     compression_ratio = calculate_compression_ratio(doc, token_ids)
    #     ts_compression_ratios.append(compression_ratio)
        
    #     print(f"文档 {i+1}:")
    #     print(f"  原始字节数: {len(doc.encode('utf-8'))}")
    #     print(f"  令牌数: {len(token_ids)}")
    #     print(f"  压缩比: {compression_ratio:.2f} bytes/token")
    #     print(f"  文档预览: {doc[:100]}...")
    #     print()
    
    # # 分析OpenWebText文档
    # print("\n=== OpenWebText文档分析 (使用32K词汇tokenizer) ===")
    # owt_compression_ratios = []
    
    # for i, doc in enumerate(owt_docs):
    #     # 使用OpenWebText tokenizer编码
    #     token_ids = owt_tokenizer.encode(doc)
    #     compression_ratio = calculate_compression_ratio(doc, token_ids)
    #     owt_compression_ratios.append(compression_ratio)
        
    #     print(f"文档 {i+1}:")
    #     print(f"  原始字节数: {len(doc.encode('utf-8'))}")
    #     print(f"  令牌数: {len(token_ids)}")
    #     print(f"  压缩比: {compression_ratio:.2f} bytes/token")
    #     print(f"  文档预览: {doc[:100]}...")
    #     print()
    
    # # 计算平均压缩比
    # avg_ts_compression = sum(ts_compression_ratios) / len(ts_compression_ratios)
    # avg_owt_compression = sum(owt_compression_ratios) / len(owt_compression_ratios)
    
    # print("=== 总结 ===")
    # print(f"TinyStories tokenizer (10K词汇) 平均压缩比: {avg_ts_compression:.2f} bytes/token")
    # print(f"OpenWebText tokenizer (32K词汇) 平均压缩比: {avg_owt_compression:.2f} bytes/token")
    
    # # 交叉测试：用TinyStories tokenizer编码OpenWebText文档
    # print("\n=== 交叉测试：用TinyStories tokenizer编码OpenWebText文档 ===")
    # ts_on_owt_ratios = []
    # for i, doc in enumerate(owt_docs):
    #     token_ids = ts_tokenizer.encode(doc)
    #     compression_ratio = calculate_compression_ratio(doc, token_ids)
    #     ts_on_owt_ratios.append(compression_ratio)
    #     print(f"文档 {i+1}: {compression_ratio:.2f} bytes/token")
    
    # avg_ts_on_owt = sum(ts_on_owt_ratios) / len(ts_on_owt_ratios)
    # print(f"TinyStories tokenizer在OpenWebText文档上的平均压缩比: {avg_ts_on_owt:.2f} bytes/token")
    
    # # 交叉测试：用OpenWebText tokenizer编码TinyStories文档
    # print("\n=== 交叉测试：用OpenWebText tokenizer编码TinyStories文档 ===")
    # owt_on_ts_ratios = []
    # for i, doc in enumerate(ts_docs):
    #     token_ids = owt_tokenizer.encode(doc)
    #     compression_ratio = calculate_compression_ratio(doc, token_ids)
    #     owt_on_ts_ratios.append(compression_ratio)
    #     print(f"文档 {i+1}: {compression_ratio:.2f} bytes/token")
    
    # avg_owt_on_ts = sum(owt_on_ts_ratios) / len(owt_on_ts_ratios)
    # print(f"OpenWebText tokenizer在TinyStories文档上的平均压缩比: {avg_owt_on_ts:.2f} bytes/token")
    
    print(f"采样了 {len(ts_docs)} 个TinyStories文档")
    print(f"采样了 {len(owt_docs)} 个OpenWebText文档")

    # 吞吐量基准（对 encode 进行计时）
    benchmark_encode(ts_tokenizer, ts_docs, "TinyStories tokenizer 在 TinyStories 文档")
    benchmark_encode(owt_tokenizer, owt_docs, "OpenWebText tokenizer 在 OpenWebText 文档")

if __name__ == "__main__":
    main()
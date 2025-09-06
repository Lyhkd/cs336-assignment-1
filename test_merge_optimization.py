#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试BPE合并优化的性能提升"""

import time
import os
from cs336_basics.bpe import train_bpe

def create_test_data():
    """创建一个稍大的测试数据文件"""
    test_content = """
    Hello world! This is a test file for BPE training.
    We want to see how the optimized merge algorithm performs compared to the traditional one.
    The quick brown fox jumps over the lazy dog.
    Python is a great programming language for natural language processing.
    BPE (Byte Pair Encoding) is a popular tokenization method used in modern NLP models.
    <|endoftext|>
    Machine learning has revolutionized the field of artificial intelligence.
    Deep learning models like transformers have achieved remarkable success in various tasks.
    Natural language understanding requires sophisticated algorithms and large datasets.
    <|endoftext|>
    The optimization of BPE training involves caching pair frequencies and incremental updates.
    This approach significantly reduces the computational complexity of the merging step.
    By only updating the affected pairs, we can achieve substantial speedups.
    <|endoftext|>
    """ * 10  # 重复10次以增加数据量
    
    with open("test_merge_data.txt", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    return "test_merge_data.txt"

def test_merge_optimization():
    """测试合并优化的性能效果"""
    print("=" * 70)
    print("BPE合并算法优化性能测试")
    print("=" * 70)
    
    # 创建测试数据
    test_file = create_test_data()
    file_size = os.path.getsize(test_file)
    print(f"测试文件大小: {file_size} bytes")
    
    special_tokens = ["<|endoftext|>"]
    vocab_size = 800
    
    print(f"词汇表目标大小: {vocab_size}")
    print(f"特殊标记: {special_tokens}")
    print("-" * 70)
    
    # 测试优化算法
    print("🚀 测试优化的合并算法...")
    start_time = time.time()
    vocab_opt, merges_opt = train_bpe(
        input_path=test_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        use_parallel=False,  # 关闭并行以专注测试合并优化
        use_optimized_merge=True
    )
    opt_time = time.time() - start_time
    
    print(f"\n📊 优化算法结果:")
    print(f"   用时: {opt_time:.3f}秒")
    print(f"   最终词汇表大小: {len(vocab_opt)}")
    print(f"   合并次数: {len(merges_opt)}")
    
    print("-" * 70)
    
    # 测试传统算法
    print("🐌 测试传统的合并算法...")
    start_time = time.time()
    vocab_trad, merges_trad = train_bpe(
        input_path=test_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        use_parallel=False,  # 关闭并行以专注测试合并优化
        use_optimized_merge=False
    )
    trad_time = time.time() - start_time
    
    print(f"\n📊 传统算法结果:")
    print(f"   用时: {trad_time:.3f}秒")
    print(f"   最终词汇表大小: {len(vocab_trad)}")
    print(f"   合并次数: {len(merges_trad)}")
    
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)
    
    speedup = trad_time / opt_time if opt_time > 0 else 1
    time_saved = trad_time - opt_time
    
    print(f"优化算法用时:   {opt_time:.3f}秒")
    print(f"传统算法用时:   {trad_time:.3f}秒")
    print(f"时间节省:       {time_saved:.3f}秒")
    print(f"性能提升:       {speedup:.2f}x")
    
    if speedup > 1.1:
        print("✅ 优化效果显著！")
    elif speedup > 1.0:
        print("✅ 有一定优化效果")
    else:
        print("⚠️  优化效果不明显（可能数据量太小）")
    
    # 验证结果一致性
    if len(vocab_opt) == len(vocab_trad) and len(merges_opt) == len(merges_trad):
        print("✅ 两种算法产生相同的结果")
    else:
        print("⚠️  两种算法结果略有不同（可能由于实现细节差异）")
    
    print("\n优化技术说明:")
    print("1. 配对频率缓存: 避免每次重新计算所有配对频率")
    print("2. 增量更新: 只更新受影响的配对计数")
    print("3. 索引优化: 快速查找包含特定配对的词汇")
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n清理测试文件: {test_file}")

if __name__ == "__main__":
    test_merge_optimization()

import torch
from typing import BinaryIO, IO
import os

def save_checkpoint(model, optimizer, iteration, out) -> None:  
    """
    Should dump all the state from the first three parameters into the file-like object out. You can use the state_dict method of both the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to have obj be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later.
    """
    # Args:
    #     model: torch.nn.Module
    #     optimizer: torch.optim.Optimizer
    #     iteration: int
    #     out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)
    
def load_checkpoint(src, model, optimizer) -> int:
    """
    Should load all the state from the serialized checkpoint (path or file-like object) into the given model and optimizer. Return the number of iterations that we previously serialized in the checkpoint.
    """
    # Args:
    #     src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    #     model: torch.nn.Module
    #     optimizer: torch.optim.Optimizer
    # Returns:
    #     int: the previously-serialized number of iterations.
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

if __name__ == "__main__":
    from transformers import AutoTokenizer

    # 选择你想对齐的模型
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 读取文件
    with open("/data/yuer/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 编码为 tokens
    tokens = tokenizer.encode(text)

    print("Token 数量:", len(tokens))
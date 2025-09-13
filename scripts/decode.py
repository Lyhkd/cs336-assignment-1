from cs336_basics.transformer import Transformer_LM
from cs336_basics.utils import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.optimizer import AdamW
import json
config = "config/debug.json"
with open(config, "r") as f:
    config = json.load(f)
    
model = Transformer_LM.from_config(config)
tokenizer = Tokenizer.from_files("/data/yuer/assignment1-basics/data/TinyStories/vocab_str.json", "/data/yuer/assignment1-basics/data/TinyStories/merges_str.json", ["<|endoftext|>"])

# prompt = "Once upon a time, there was a boy who lived in a small village. He was a very curious boy and he loved to explore the world. One day, he decided to go on a journey to find a new place to live. He packed his bags and set off on his journey. He walked for days and days and finally he reached a beautiful place. He built a house there and lived happily ever after."
prompt = "Once upon a time, there was a pretty girl named Lily."
optimizer = AdamW(model.parameters(), lr=config["max_learning_rate"], weight_decay=config["weight_decay"])

load_checkpoint("checkpoints/lr1e3/checkpoint_final.pt", model, optimizer)

print(model.decode(tokenizer, prompt, max_new_tokens=256, temperature=1.5, top_p=0.95))

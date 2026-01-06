from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("Nan-Do/code-search-net-python", split="train", streaming=True)
print("Success!")
print(next(iter(dataset)))

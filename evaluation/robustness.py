import random

def perturb_docstring(docstring):
    """
    Perturbs the docstring to test model robustness.
    Strategies:
    1. Add a prefix.
    2. Change casing of some words.
    """
    if not docstring:
        return ""
    
    # Strategy 1: Add prefix
    prefixes = [
        "Implement this function: ",
        "Write a python function that ",
        "Coding task: ",
        "Please help me with this: "
    ]
    prefix = random.choice(prefixes)
    
    # Strategy 2: Randomly uppercase some words
    words = docstring.split()
    perturbed_words = [w.upper() if random.random() < 0.1 else w for w in words]
    
    return prefix + " ".join(perturbed_words)

from datasketches import update_theta_sketch, theta_jaccard_similarity

def calculate_minhash_similarity(text1, text2):
    """
    Calculates Jaccard similarity between two texts using Theta Sketch (MinHash).
    """
    if not text1 or not text2:
        return 0.0
        
    # Simple tokenization by whitespace
    tokens1 = text1.split()
    tokens2 = text2.split()
    
    sketch1 = update_theta_sketch()
    for t in tokens1:
        sketch1.update(t)
        
    sketch2 = update_theta_sketch()
    for t in tokens2:
        sketch2.update(t)
        
    return theta_jaccard_similarity.jaccard(sketch1, sketch2)[1]

def check_exact_match(text1, text2):
    """
    Checks if two texts are exactly the same (ignoring leading/trailing whitespace).
    """
    if not text1 or not text2:
        return False
    return text1.strip() == text2.strip()

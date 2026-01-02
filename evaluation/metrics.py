from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def is_memorized(pred, gold, threshold=0.7):
    score = scorer.score(gold, pred)["rougeL"].fmeasure
    return score >= threshold

import pandas as pd
import evaluate
import numpy as np

df = pd.read_csv("data/blip2_results.csv")

df["generated_answer"] = df["generated_answer"].fillna("")
references = df["answer"].tolist()
predictions = df["generated_answer"].tolist()

print("Number of data points:", len(predictions))

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

references_for_bleu = [[ref] for ref in references]

bertscore_result = bertscore.compute(
    predictions=predictions, 
    references=references,
    model_type="bert-base-uncased"
)
mean_f1 = np.mean(bertscore_result["f1"])
mean_precision = np.mean(bertscore_result["precision"])
mean_recall = np.mean(bertscore_result["recall"])
print("BERTScore F1:", mean_f1)
print("BERTScore Precision:", mean_precision)
print("BERTScore Recall:", mean_recall)

bleu_result = bleu.compute(
    predictions=predictions, 
    references=references_for_bleu
)
print("BLEU:", bleu_result['bleu'])

rouge_result = rouge.compute(
    predictions=predictions, 
    references=references
)
print("ROUGE:", rouge_result['rougeL'])

meteor_result = meteor.compute(
    predictions=predictions,
    references=references
)
print("METEOR:", meteor_result['meteor'])

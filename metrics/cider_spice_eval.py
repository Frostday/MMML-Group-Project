import csv
import os
import json
import zipfile
from bert_score import BERTScorer
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

def save_result(result, result_dir, filename, remove_duplicate='', is_gt=False):
    final_result_file = os.path.join(result_dir, f'{filename}.json')

    if remove_duplicate:
        result_new = []
        id_list = []
        for res in result:
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new

    if is_gt:
        images = [{"id": res["id"]} for res in result]
        result = {"annotations": result, "images": images}

    with open(final_result_file, 'w') as f:
        json.dump(result, f)
    return final_result_file

def _evaluate_common(pred_list, gt_list, scorer, result_tag):
    pred_file = save_result(pred_list, '.', f'{result_tag}_pred')
    gt_file = save_result(gt_list, '.', f'{result_tag}_gt', is_gt=True)

    coco = COCO(gt_file)
    coco_result = coco.loadRes(pred_file)

    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    results = {}
    for metric in ['CIDEr', 'SPICE']:
        results[metric] = coco_eval.eval.get(metric, 0.0)

    for metric in results:
        print(f'{metric}: {results[metric]:.3f}')

    return results, coco_eval.eval

def evaluate_csv(csv_path, model_name="baseline", limit=None):
    scorer = BERTScorer(model_type='bert-base-uncased')
    pred_list, gt_list = [], []
    BERTScore_F1 = 0
    failed_parsing = 0
    no_samples = 0
    all = 0

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            gt = row['answer'].strip()
            generated = row['generated_answer'].strip()

            if not generated:
                no_samples += 1
                generated = ""

            try:
                _, _, F1 = scorer.score([generated], [gt])
                BERTScore_F1 += F1
            except:
                failed_parsing += 1

            pred_list.append({"image_id": i, "caption": generated})
            gt_list.append({"image_id": i, "id": i, "caption": gt})
            all += 1

    print(f"\n📄 Evaluating CSV: {csv_path} (model: {model_name})")
    results, _ = _evaluate_common(pred_list, gt_list, scorer, f'{model_name}_csv')
    if all > 0:
        score = BERTScore_F1 / all
        score = score.item() if hasattr(score, "item") else score
        print(f'BERTScore F1: {score:.3f}')
    else:
        print("BERTScore F1: N/A")

    print(f'Failed parsing: {failed_parsing}')
    print(f'Samples evaluated: {all}')
    print(f'No-answer samples: {no_samples}\n')


def evaluate_zip(zip_path, model_name="baseline", limit=None):
    scorer = BERTScorer(model_type='bert-base-uncased')
    pred_list, gt_list = [], []
    BERTScore_F1 = 0
    failed_parsing = 0
    no_samples = 0
    counter = 0
    all = 0

    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if f.endswith('.json')]
        for fname in files:
            with z.open(fname) as f:
                data = json.load(f)

            for _, sample in data.items():
                gt = sample["answer"]
                responses = sample["response"]
                added = False

                for _, response_list in responses.items():
                    if "no" in response_list[0].lower():
                        no_samples += 1
                        continue

                    generated = response_list[1]
                    try:
                        _, _, F1 = scorer.score([generated], [gt])
                        BERTScore_F1 += F1
                    except:
                        failed_parsing += 1

                    pred_list.append({"image_id": counter, "caption": generated})
                    gt_list.append({"image_id": counter, "id": counter, "caption": gt})
                    counter += 1
                    all += 1
                    added = True

                    if limit is not None and all >= limit:
                        break

                if not added:
                    pred_list.append({"image_id": counter, "caption": ""})
                    gt_list.append({"image_id": counter, "id": counter, "caption": gt})
                    counter += 1
                    all += 1

                if limit is not None and all >= limit:
                    break
            if limit is not None and all >= limit:
                break

    print(f"\n📦 Evaluating ZIP: {zip_path} (model: {model_name})")
    results, _ = _evaluate_common(pred_list, gt_list, scorer, f'{model_name}_zip')
    if all > 0:
        score = BERTScore_F1 / all
        score = score.item() if hasattr(score, "item") else score
        print(f'BERTScore F1: {score:.3f}')
    else:
        print("BERTScore F1: N/A")
    print(f'Failed parsing: {failed_parsing}')
    print(f'Samples evaluated: {all}')
    print(f'No-answer samples: {no_samples}\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model outputs using CIDEr, SPICE, and BERTScore")
    parser.add_argument("--csv", type=str, help="Path to CSV file (use for CSV-based predictions)")
    parser.add_argument("--zip", type=str, help="Path to ZIP file (use for SPIQA-style JSON responses)")
    parser.add_argument("--model_name", type=str, default="baseline", help="Name of the model being evaluated")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (default: all)")

    args = parser.parse_args()

    if args.csv:
        evaluate_csv(args.csv, model_name=args.model_name, limit=args.limit)
    elif args.zip:
        evaluate_zip(args.zip, model_name=args.model_name, limit=args.limit)
    else:
        print("Please provide either --csv or --zip input file.")


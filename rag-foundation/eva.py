import argparse
import json
import re
import string
from collections import Counter

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def token_f1_score(prediction, ground_truth):
    """
    Calculate F1 score based on token overlap.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def paragraph_f1_score(prediction, ground_truth):
    """
    Calculate F1 score based on overlap of sets.
    """
    if not ground_truth and not prediction:
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def lcs_length(x, y):
    """
    Compute the length of the longest common subsequence between two strings.
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def rouge_l_score(prediction, ground_truth):
    """
    Calculate ROUGE-L score.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    lcs_len = lcs_length(pred_tokens, gt_tokens)
    if not gt_tokens:
        return 0.0
    return lcs_len / len(gt_tokens)

def f1_score(prediction, ground_truth):
    """
    Calculate F1 score based on exact matches.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not gt_tokens:
        return 0.0
    correct = len(set(pred_tokens) & set(gt_tokens))
    precision = correct / len(pred_tokens) if pred_tokens else 0
    recall = correct / len(gt_tokens) if gt_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append(
                        {"answer": "Unanswerable", "evidence": [], "type": "none"}
                    )
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(
                            f"Annotation {answer_info['annotation_id']} does not contain an answer"
                        )
                    if text_evidence_only:
                        evidence = [
                            text
                            for text in answer_info["evidence"]
                            if "FLOAT SELECTED" not in text
                        ]
                    else:
                        evidence = answer_info["evidence"]
                    references.append(
                        {"answer": answer, "evidence": evidence, "type": answer_type}
                    )
            answers_and_evidence[question_id] = references

    return answers_and_evidence

def evaluate(gold, predicted, retrieval_only=False):
    max_answer_f1s = []
    max_evidence_f1s = []
    max_rouge_l_scores = []
    max_f1_scores = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            max_rouge_l_scores.append(0.0)
            max_f1_scores.append(0.0)
            continue
        answer_f1s_and_types = [
            (
                token_f1_score(predicted[question_id]["answer"], reference["answer"]),
                reference["type"],
            )
            for reference in gold[question_id]
        ]
        max_answer_f1, answer_type = sorted(
            answer_f1s_and_types, key=lambda x: x[0], reverse=True
        )[0]
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)
        evidence_f1s = [
            paragraph_f1_score(
                predicted[question_id]["evidence"], reference["evidence"]
            )
            for reference in gold[question_id]
        ]
        max_evidence_f1s.append(max(evidence_f1s))
        rouge_l_scores = [
            rouge_l_score(
                predicted[question_id]["answer"], reference["answer"]
            )
            for reference in gold[question_id]
        ]
        max_rouge_l_scores.append(max(rouge_l_scores))
        f1_scores = [
            f1_score(
                predicted[question_id]["answer"], reference["answer"]
            )
            for reference in gold[question_id]
        ]
        max_f1_scores.append(max(f1_scores))

    mean = lambda x: sum(x) / len(x) if x else 0.0

    if not retrieval_only:
        return {
            "Answer F1": mean(max_answer_f1s),
            "Answer F1 by type": {
                key: mean(value) for key, value in max_answer_f1s_by_type.items()
            },
            "Evidence F1": mean(max_evidence_f1s),
            "ROUGE-L": mean(max_rouge_l_scores),
            "F1 Score": mean(max_f1_scores),
            "Missing predictions": num_missing_predictions,
        }
    else:
        return {
            "Evidence F1": mean(max_evidence_f1s),
            "ROUGE-L": mean(max_rouge_l_scores),
            "F1 Score": mean(max_f1_scores),
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}""",
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset",
    )
    parser.add_argument(
        "--retrieval_only",
        help="If set, the evaluator will just evaluate the retrieval scores",
        action="store_true",
    )
    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1",
    )
    args = parser.parse_args()
    gold_data = json.load(open(args.gold))
    gold_answers_and_evidence = get_answers_and_evidence(
        gold_data, args.text_evidence_only
    )
    predictions_data = [
        json.loads(line) for line in open(args.predictions, "r")
    ]
    predicted_answers_and_evidence = {
        entry["question_id"]: {
            "answer": entry["predicted_answer"],
            "evidence": entry["predicted_evidence"],
        }
        for entry in predictions_data
    }
    results = evaluate(
        gold_answers_and_evidence,
        predicted_answers_and_evidence,
        args.retrieval_only,
    )
    print(json.dumps(results, indent=2))

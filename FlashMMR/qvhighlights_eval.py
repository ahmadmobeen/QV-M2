import json
from standalone_eval.eval import eval_submission, load_jsonl

class QVHighlightsEval:
    def __init__(self, gt_path):
        """
        Wrapper to allow FlashMMR/train.py to use the standalone_eval logic.
        """
        self.gt_path = gt_path
        self.ground_truth = load_jsonl(gt_path)

    def eval(self, submission):
        """
        Args:
            submission: list of dicts with [qid, query, vid, pred_relevant_windows, pred_saliency_scores]
        """
        # The standalone evaluator expects specific keys to exist.
        # If train.py passes a filename instead of a list, load it.
        if isinstance(submission, str):
            submission = load_jsonl(submission)
            
        results = eval_submission(submission, self.ground_truth, verbose=False)
        return results

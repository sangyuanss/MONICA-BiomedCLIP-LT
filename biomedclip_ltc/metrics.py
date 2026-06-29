"""Head/Medium/Tail metrics — reuses MONICA's utils/log_accuracy.

calculate_metrics computes per-class + grouped Acc / AUROC / AUPRC / F1 with the
head/medium cutoffs from the SAME config the baselines use, so the numbers are
identical in definition to MONICA. Selection metric = group-average accuracy
(`accuracy[3]`), matching MONICA's best.pt criterion.
"""
import torch

from utils.log_accuracy import calculate_metrics as _monica_metrics


def evaluate_logits(cfg, logits, labels):
    """logits: [N,C] tensor, labels: [N] tensor (CPU). Returns MONICA metric dict."""
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    return _monica_metrics(cfg, [logits], [labels])


def group_avg_acc(results):
    """Group-average accuracy = mean(head, medium, tail) — the selection metric."""
    return float(results["accuracy"][3])


def format_summary(results, tag=""):
    """One-line human summary: overall(group-avg) + Head/Med/Tail acc."""
    h, m, t, avg = results["accuracy"]
    _, _, _, aavg = results["aucs"]
    _, _, _, pavg = results["auprcs"]
    _, _, _, favg = results["f1s"]
    return (f"{tag} groupAvgAcc={avg:.2f} | Head={h:.2f} Med={m:.2f} Tail={t:.2f} "
            f"| AUROC={100*aavg:.2f} AUPRC={100*pavg:.2f} F1={100*favg:.2f}")


def results_to_jsonable(results):
    """Convert numpy arrays in a metric dict to plain lists for JSON dumping."""
    out = {}
    for k, v in results.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) for x in v]
        else:
            out[k] = v
    return out

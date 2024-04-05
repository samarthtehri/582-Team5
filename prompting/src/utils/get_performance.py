import sklearn.metrics


def get_performance(y_true: list[int], y_pred: list[int]) -> dict:
    # statistics
    statistics = {
        "total_samples": len(y_true),
        "label_distribution": {
            "count": {"0": y_true.count(0), "1": y_true.count(1)},
            "average": sum(y_true) / len(y_true),
        },
    }
    
    # performance
    performance = {}
    performance["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)
    for pos_label in [0, 1]:
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, # pos_label=pos_label,
            zero_division=0,
            labels=[pos_label],
        )

        performance[f"pos_label={pos_label}"] = {
            "precision": precision[0].item(),
            "recall": recall[0].item(),
            "f1": f1[0].item(),
        }
    
    return {
        "statistics": statistics,
        "performance": performance,
    }

import sklearn.metrics


def get_label_distrubiton(labels_list: list[int]) -> dict:
    labels_list = [int(label) for label in labels_list]  # make sure all labels are integers
    
    return {
        "count": {"0": labels_list.count(0), "1": labels_list.count(1), "-1": labels_list.count(-1)},
        "average": sum(labels_list) / len(labels_list),
    }


def get_performance(y_true: list[int], y_pred: list[int]) -> dict:
    # statistics
    statistics = {
        "total_samples": len(y_true),
        "y_true_distribution": get_label_distrubiton(y_true),
        "y_pred_distribution": get_label_distrubiton(y_pred),
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
    
    # overall
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred,
        zero_division=0,
        average="macro", labels=[0, 1],
    )
    
    performance["macro"] = {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }
    
    return {
        "statistics": statistics,
        "performance": performance,
    }

from sklearn.metrics import roc_auc_score
from MLstatkit.stats import Delong_test


def delong_test(method_results):

    comparison_results = []
    model_names = list(method_results.keys())

    # Delong's test
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]

            try:
                # Get aggregated predictions and true labels for each model
                y_true = method_results[model1]["y_true"]
                preds1 = method_results[model1]["y_pred"]
                preds2 = method_results[model2]["y_pred"]

                auc_model1 = roc_auc_score(y_true, preds1)
                auc_model2 = roc_auc_score(y_true, preds2)

                z, p = Delong_test(y_true, preds1, preds2)
                comparison_results.append((model1, model2, auc_model1, auc_model2, z, p))

            except Exception as e:
                print(f"Error in DeLong's test comparing {model1.upper()} and {model2.upper()}: {e}")

    return comparison_results
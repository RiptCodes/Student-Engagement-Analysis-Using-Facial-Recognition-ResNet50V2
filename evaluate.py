import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve,
                              f1_score, precision_score, recall_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from config import *
from dataset import DataGenerator

os.makedirs('outputs/plots', exist_ok=True)


def save_fig(name):
    path = f'outputs/plots/{name}.png'
    counter = 1
    while os.path.exists(path):
        path = f'outputs/plots/{name}_{counter}.png'
        counter += 1
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def evaluate(test_gen):
    model     = tf.keras.models.load_model(MODEL_PATH)
    test_ds   = test_gen.as_tf_dataset(shuffle=False)

    all_preds = []
    all_true  = []
    all_probs = []

    for X, y in test_ds:
        probs = model(X, training=False).numpy()
        all_probs.extend(probs)
        all_preds.extend(np.argmax(probs, axis=1))
        all_true.extend(np.argmax(y.numpy(), axis=1))

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    all_probs = np.array(all_probs)

    print(f'Test Accuracy: {np.mean(all_preds == all_true):.4f}')
    print(classification_report(all_true, all_preds, target_names=LABELS, zero_division=0))

    return model, all_preds, all_true, all_probs


def plot_confusion(all_true, all_preds):
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    save_fig('confusion_matrix')


def plot_roc_curve(all_true, all_probs):
    fpr, tpr, _ = roc_curve(all_true, all_probs[:, 1])
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', linewidth=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    save_fig('roc_curve')
    return roc_auc


def plot_precision_recall(all_true, all_probs):
    precision, recall, _ = precision_recall_curve(all_true, all_probs[:, 1])
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', linewidth=2,
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    save_fig('precision_recall_curve')
    return pr_auc


def plot_per_class_metrics(all_true, all_preds):
    precision = precision_score(all_true, all_preds, average=None, zero_division=0)
    recall    = recall_score(all_true, all_preds, average=None, zero_division=0)
    f1        = f1_score(all_true, all_preds, average=None, zero_division=0)

    x     = np.arange(len(LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width, precision, width, label='Precision', color='steelblue',  alpha=0.8)
    ax.bar(x,         recall,    width, label='Recall',    color='darkorange', alpha=0.8)
    ax.bar(x + width, f1,        width, label='F1 Score',  color='green',      alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.legend()

    for i in range(len(LABELS)):
        ax.text(i - width, precision[i] + 0.02, f'{precision[i]:.2f}', ha='center', fontsize=8)
        ax.text(i,         recall[i]    + 0.02, f'{recall[i]:.2f}',    ha='center', fontsize=8)
        ax.text(i + width, f1[i]        + 0.02, f'{f1[i]:.2f}',        ha='center', fontsize=8)

    plt.tight_layout()
    save_fig('per_class_metrics')


def plot_threshold_analysis(all_true, all_probs):
    thresholds = np.arange(0.1, 0.9, 0.05)
    precisions = []
    recalls    = []
    f1s        = []

    for thresh in thresholds:
        preds = (all_probs[:, 1] >= thresh).astype(int)
        precisions.append(precision_score(all_true, preds, zero_division=0))
        recalls.append(recall_score(all_true, preds, zero_division=0))
        f1s.append(f1_score(all_true, preds, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='steelblue',  linewidth=2)
    plt.plot(thresholds, recalls,    label='Recall',    color='darkorange', linewidth=2)
    plt.plot(thresholds, f1s,        label='F1',        color='green',      linewidth=2)
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default threshold (0.5)')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall and F1 vs Decision Threshold')
    plt.legend()
    plt.tight_layout()
    save_fig('threshold_analysis')


def plot_confidence(model, test_gen, all_true):
    all_confs   = []
    all_correct = []

    for X, y in test_gen.as_tf_dataset(shuffle=False):
        preds = model(X, training=False).numpy()
        for j in range(len(preds)):
            all_confs.append(float(np.max(preds[j])))
            all_correct.append(np.argmax(preds[j]) == np.argmax(y.numpy()[j]))

    all_confs   = np.array(all_confs)
    all_correct = np.array(all_correct)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(all_confs[all_correct],  bins=20, alpha=0.7, label='Correct',   color='green')
    ax1.hist(all_confs[~all_correct], bins=20, alpha=0.7, label='Incorrect', color='red')
    ax1.set_title('Confidence Distribution')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.legend()

    for i, label in enumerate(LABELS):
        mask = all_true == i
        if mask.sum() > 0:
            ax2.bar(label, np.mean(all_confs[mask]), alpha=0.8)
    ax2.set_title('Mean Confidence per Class')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    save_fig('confidence')


def plot_sample_predictions(model, test_gen, all_true, all_preds, n=12):
    samples_correct   = np.where(all_true == all_preds)[0][:n//2]
    samples_incorrect = np.where(all_true != all_preds)[0][:n//2]
    sample_indices    = set(np.concatenate([samples_correct, samples_incorrect]))

    frames = []
    labels = []
    count  = 0

    for X, y in test_gen.as_tf_dataset(shuffle=False):
        for j in range(len(X)):
            if count in sample_indices:
                frames.append(X[j].numpy())
                labels.append(np.argmax(y.numpy()[j]))
            count += 1
            if len(frames) == len(sample_indices):
                break
        if len(frames) == len(sample_indices):
            break

    cols = 4
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, (frame, true_label) in enumerate(zip(frames, labels)):
        img = frame.copy()
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        img = np.clip(img, 0, 1)

        pred_label = all_preds[list(sample_indices)[i]]
        correct    = true_label == pred_label
        colour     = 'green' if correct else 'red'

        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {LABELS[true_label]}\nPred: {LABELS[pred_label]}",
            fontsize=8, color=colour
        )
        axes[i].axis('off')

    for i in range(len(frames), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions (green = correct, red = incorrect)', fontsize=11)
    plt.tight_layout()
    save_fig('sample_predictions')


def baseline_comparison(model, test_gen, all_true, all_preds):
    print("\nExtracting features for baseline comparison...")

    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-4].output
    )

    features = []
    for X, y in test_gen.as_tf_dataset(shuffle=False):
        feats = feature_model(X, training=False).numpy()
        features.extend(feats)

    features = np.array(features)
    split    = int(len(features) * 0.7)
    X_train  = features[:split]
    X_test   = features[split:]
    y_train  = all_true[:split]
    y_test   = all_true[split:]

    results = {
        'ResNet50V2': {
            'acc':    np.mean(all_preds == all_true),
            'f1_not': f1_score(all_true, all_preds, average=None, zero_division=0)[0],
            'f1_eng': f1_score(all_true, all_preds, average=None, zero_division=0)[1],
        }
    }

    print("Training SVM...")
    svm       = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    results['SVM'] = {
        'acc':    np.mean(svm_preds == y_test),
        'f1_not': f1_score(y_test, svm_preds, average=None, zero_division=0)[0],
        'f1_eng': f1_score(y_test, svm_preds, average=None, zero_division=0)[1],
    }

    print("Training Random Forest...")
    rf       = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results['Random Forest'] = {
        'acc':    np.mean(rf_preds == y_test),
        'f1_not': f1_score(y_test, rf_preds, average=None, zero_division=0)[0],
        'f1_eng': f1_score(y_test, rf_preds, average=None, zero_division=0)[1],
    }

    print("\n" + "="*60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Not Eng':>12} {'F1 Engaged':>12}")
    print("="*60)
    for name, r in results.items():
        print(f"{name:<25} {r['acc']:>10.4f} {r['f1_not']:>12.4f} {r['f1_eng']:>12.4f}")
    print("="*60)

    models     = list(results.keys())
    accuracies = [results[m]['acc']    for m in models]
    f1_not     = [results[m]['f1_not'] for m in models]
    f1_eng     = [results[m]['f1_eng'] for m in models]

    x     = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, accuracies, width, label='Accuracy',       color='steelblue',  alpha=0.8)
    ax.bar(x,         f1_not,     width, label='F1 Not Engaged', color='red',        alpha=0.8)
    ax.bar(x + width, f1_eng,     width, label='F1 Engaged',     color='green',      alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — ResNet50V2 vs Baselines')
    ax.legend()
    plt.tight_layout()
    save_fig('model_comparison')

    return results


if __name__ == '__main__':
    from preprocessing import load_labels

    _, _, test_df = load_labels()
    test_labels   = test_df['Label'].values
    test_gen      = DataGenerator(f'{SAVE_DIR}/test.tfrecord', test_labels)

    model, all_preds, all_true, all_probs = evaluate(test_gen)

    plot_confusion(all_true, all_preds)
    plot_roc_curve(all_true, all_probs)
    plot_precision_recall(all_true, all_probs)
    plot_per_class_metrics(all_true, all_preds)
    plot_threshold_analysis(all_true, all_probs)
    plot_confidence(model, test_gen, all_true)
    plot_sample_predictions(model, test_gen, all_true, all_preds)
    baseline_comparison(model, test_gen, all_true, all_preds)

    print("\nAll plots saved to outputs/plots/")
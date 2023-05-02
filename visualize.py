import matplotlib
import matplotlib.pyplot as plt
import os, json

diseases = ["Lung opacity", "Pleural effusion", "Atelectasis", "Enlarged cardiac silhouette", "Pulmonary edema/hazy opacity", 
            "Pneumothorax", "Consolidation" , "Fluid overload/heart failure", "Pneumonia"]


def plot_roc(metrics, output_dir, filename, labels=diseases):
    fig, axs = plt.subplots(2, len(labels), figsize=(24,12))

    for i, (fpr, tpr, aucs, precision, recall, label) in enumerate(zip(metrics['fpr'].values(), metrics['tpr'].values(),
                                                                       metrics['aucs'].values(), metrics['precision'].values(),
                                                                       metrics['recall'].values(), labels)):
        # top row -- ROC
        axs[0,i].plot(fpr, tpr, label='AUC = %0.2f' % aucs)
        axs[0,i].plot([0, 1], [0, 1], 'k--')  # diagonal margin
        axs[0,i].set_xlabel('False Positive Rate')
        # bottom row - Precision-Recall
        axs[1,i].step(recall, precision, where='post')
        axs[1,i].set_xlabel('Recall')
        # format
        axs[0,i].set_title(label)
        axs[0,i].legend(loc="lower right")

    plt.suptitle(filename)
    axs[0,0].set_ylabel('True Positive Rate')
    axs[1,0].set_ylabel('Precision')

    for ax in axs.flatten():
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', filename + '.png'), pad_inches=0.)
    plt.close()

if __name__=="__main__":
    metrics_path = "/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/eval_metrics(2).json"
    output_dir = "/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/"
    filename = "roc"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    plot_roc(metrics, output_dir, filename)
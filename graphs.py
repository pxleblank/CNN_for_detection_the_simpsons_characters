from imports import *


def show_graphs(directory=None):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train_loss', 'val_loss'])
    plt.show()
    # plt.savefig(directory + "/" + "loss", bbox_inches='tight')

    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.show()

    plt.plot(val_precision)
    plt.plot(val_recall)
    plt.legend(['val_precision', 'val_recall'])
    plt.show()

    plt.plot(lr_list)
    plt.legend(['learning_rate'])
    plt.show()

    # bar plot for precision
    plt.figure(figsize=(10, 6))
    precision_values = val_precision_per_epoch[-1]  # Значения precision на последней эпохе
    x = np.arange(num_classes)  # Индексы классов
    plt.bar(x, precision_values, color='skyblue', alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.title("Precision for each class (Last Epoch)")
    plt.xticks(x, [f'Class {i}' for i in range(num_classes)], rotation=45)
    plt.tight_layout()
    plt.show()

    # bar plot for recall
    plt.figure(figsize=(10, 6))
    recall_values = val_recall_per_epoch[-1]  # Значения recall на последней эпохе
    plt.bar(x, recall_values, color='salmon', alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.title("Recall for each class (Last Epoch)")
    plt.xticks(x, [f'Class {i}' for i in range(num_classes)], rotation=45)
    plt.tight_layout()
    plt.show()

    # Test

    # bar plot for precision
    plt.figure(figsize=(10, 6))
    precision_values = test_precision_per_epoch[-1]  # Значения precision на последней эпохе
    x = np.arange(num_classes)  # Индексы классов
    plt.bar(x, precision_values, color='skyblue', alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.title("Precision for each class (Last Epoch)")
    plt.xticks(x, [f'Class {i}' for i in range(num_classes)], rotation=45)
    plt.tight_layout()
    plt.show()

    # bar plot for recall
    plt.figure(figsize=(10, 6))
    recall_values = test_recall_per_epoch[-1]  # Значения recall на последней эпохе
    plt.bar(x, recall_values, color='salmon', alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.title("Recall for each class (Last Epoch)")
    plt.xticks(x, [f'Class {i}' for i in range(num_classes)], rotation=45)
    plt.tight_layout()
    plt.show()

from imports import *


def test_model(model, loss_func, test_loader, data_test):
    # Тест модели
    model.eval()
    true_answer = 0
    TP_per_class = [0] * num_classes
    FP_per_class = [0] * num_classes
    FN_per_class = [0] * num_classes
    inner_test_loss = []
    with torch.no_grad():
        for x, labels in test_loader:

            x, labels = x.to(device), labels.to(device)

            # Прямой проход + расчёт ошибки модели
            predict = model(x)
            loss = loss_func(predict, labels)

            inner_test_loss.append(loss.item())
            mean_test_loss = sum(inner_test_loss) / len(inner_test_loss)

            predicts = predict.argmax(dim=1)
            true_answer += (predicts == labels).sum().item()

            # Подсчет TP, FP и FN для каждого класса
            for class_idx in range(num_classes):
                TP_per_class[class_idx] += ((predicts == class_idx) & (labels == class_idx)).sum().item()
                FP_per_class[class_idx] += ((predicts == class_idx) & (labels != class_idx)).sum().item()
                FN_per_class[class_idx] += ((predicts != class_idx) & (labels == class_idx)).sum().item()

    # Расчёт значения метрик (precision и recall для каждого класса и макроусреднённых значений)
    inner_test_accuracy = true_answer / len(data_test)

    precision_per_class = [
        TP_per_class[i] / (TP_per_class[i] + FP_per_class[i]) if (TP_per_class[i] + FP_per_class[i]) > 0 else 0 for i in
        range(num_classes)
    ]
    recall_per_class = [
        TP_per_class[i] / (TP_per_class[i] + FN_per_class[i]) if (TP_per_class[i] + FN_per_class[i]) > 0 else 0 for i in
        range(num_classes)
    ]
    inner_test_precision = sum(precision_per_class) / num_classes
    inner_test_recall = sum(recall_per_class) / num_classes

    # Сохранение функции потерь и метрик
    test_loss.append(mean_test_loss)
    test_accuracy.append(inner_test_accuracy)
    test_precision.append(inner_test_precision)
    test_recall.append(inner_test_recall)
    test_precision_per_epoch.append(precision_per_class)
    test_recall_per_epoch.append(recall_per_class)

    print(
        f"test_loss = {mean_test_loss:.4f}, test_accuracy = {inner_test_accuracy:.4f}, test_precision = {inner_test_precision:.4f}, test_recall = {inner_test_recall:.4f}, lr = {lr}")

from imports import *


def train_model(model, optimizer, loss_func, lr_scheduler, checkpoint, train_loader, val_loader, data_train):
    # Цикл обучения
    for epoch in range(epochs):
        # Тренировка модели
        model.train()
        true_answer = 0
        inner_train_loss = []
        mean_train_loss = 0
        train_loop = tqdm(train_loader, leave=True)
        for x, labels in train_loop:
            x, labels = x.to(device), labels.to(device)

            # Прямой проход + расчёт ошибки модели
            predict = model(x)
            loss = loss_func(predict, labels)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()

            # Шаг оптимизации
            optimizer.step()

            inner_train_loss.append(loss.item())
            mean_train_loss = sum(inner_train_loss) / len(inner_train_loss)

            true_answer += (predict.argmax(dim=1) == labels).sum().item()

            train_loop.set_description(f"Epoch ({epoch + 1}/{epochs}), train_loss = {mean_train_loss:.4f}")

        # Расчёт значения метрики
        inner_train_accuracy = true_answer / (len(data_train) * 0.8)

        # Сохранение функции потерь и метрик
        train_loss.append(mean_train_loss)
        train_accuracy.append(inner_train_accuracy)

        # Проверка модели (валидация)
        model.eval()
        true_answer = 0
        TP_per_class = [0] * num_classes
        FP_per_class = [0] * num_classes
        FN_per_class = [0] * num_classes
        inner_val_loss = []
        with torch.no_grad():
            for x, labels in val_loader:

                x, labels = x.to(device), labels.to(device)

                # Прямой проход + расчёт ошибки модели
                predict = model(x)
                loss = loss_func(predict, labels)

                inner_val_loss.append(loss.item())
                mean_val_loss = sum(inner_val_loss) / len(inner_val_loss)

                predicts = predict.argmax(dim=1)
                true_answer += (predicts == labels).sum().item()

                # Подсчет TP, FP и FN для каждого класса
                for class_idx in range(num_classes):
                    TP_per_class[class_idx] += ((predicts == class_idx) & (labels == class_idx)).sum().item()
                    FP_per_class[class_idx] += ((predicts == class_idx) & (labels != class_idx)).sum().item()
                    FN_per_class[class_idx] += ((predicts != class_idx) & (labels == class_idx)).sum().item()

        # Расчёт значения метрик (precision и recall для каждого класса и макроусреднённых значений)
        inner_val_accuracy = true_answer / (len(data_train) * 0.2)

        precision_per_class = [
            TP_per_class[i] / (TP_per_class[i] + FP_per_class[i]) if (TP_per_class[i] + FP_per_class[i]) > 0 else 0 for
            i in
            range(num_classes)
        ]
        recall_per_class = [
            TP_per_class[i] / (TP_per_class[i] + FN_per_class[i]) if (TP_per_class[i] + FN_per_class[i]) > 0 else 0 for
            i in
            range(num_classes)
        ]
        inner_val_precision = sum(precision_per_class) / num_classes
        inner_val_recall = sum(recall_per_class) / num_classes

        # Сохранение функции потерь и метрик
        val_loss.append(mean_val_loss)
        val_accuracy.append(inner_val_accuracy)
        val_precision.append(inner_val_precision)
        val_recall.append(inner_val_recall)
        val_precision_per_epoch.append(precision_per_class)
        val_recall_per_epoch.append(recall_per_class)

        # Изменение и сохранение lr
        lr_scheduler.step(inner_val_precision)
        lr = lr_scheduler._last_lr[0]
        lr_list.append(lr)

        print(
            f"Epoch ({epoch + 1}/{epochs}), train_loss = {mean_train_loss:.4f}, train_accuracy = {inner_train_accuracy:.4f}, val_loss = {mean_val_loss:.4f}, val_accuracy = {inner_val_accuracy:.4f}, val_precision = {inner_val_precision:.4f}, val_recall = {inner_val_recall:.4f}, lr = {lr}")

        # global best_loss
        # if best_loss is None:
        #     best_loss = mean_val_loss
        #
        # if mean_val_loss < best_loss - best_loss * 0.01:
        #     best_loss = mean_val_loss
        #
        #     torch.save(checkpoint, f'model_state_dict_epoch_{epoch + 1}.pt')
        #     print(
        #         f'For epoch - {epoch + 1}, the model with the value of the validation loss function is saved -- {mean_val_loss:.4f}',
        #         end='\n')

    return None

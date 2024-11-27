from imports import *


class CustomImageFolder:
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = class_to_idx

        for file_name in os.listdir(root_dir):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = root_dir + '/' + file_name

                class_name = '_'.join(file_name.split('_')[:-1])
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def show_img(train_loader):
    data_iter = iter(train_loader)  # Creating an iterator from a DataLoader
    images, labels = next(data_iter)  # Getting the first batch of images and labels
    print(labels[0])

    image = images[0]  # Extracting the first image from the batch

    # If the image has three channels (RGB), you need to change the axes for correct display
    image = image.permute(1, 2, 0)  # Changing places from (C, H, W) to (H, W, C)

    # Displaying the image
    plt.imshow(image.numpy())  # Converting tensor to numpy-array
    plt.axis('off')  # Removing the axes for a cleaner image
    plt.show()


def l1_regularization(model, lambda_l1=0.001):
    l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    return lambda_l1 * l1_norm


def l2_regularization(model, lambda_l2=0.001):
    l2_norm = sum(torch.sum(param ** 2) for param in model.parameters())
    return lambda_l2 * l2_norm



# # Specify the directory
# directory = "rename/"
# counter = 7
# # Iterate through all files in the directory
# for filename in os.listdir(directory):
#     if filename.startswith("2024"):
#         counter += 1
#         new_name = f"pic_000{counter}.png"
#         # Construct the full path for the old and new filenames
#         old_path = os.path.join(directory, filename)
#         new_path = os.path.join(directory, new_name)
#
#         # Rename the file
#         os.rename(old_path, new_path)


# # Укажите исходную и целевую папки
# source_dir = train_path + "/troy_mcclure"  # Путь к папке с исходными изображениями
# target_dir = train_path + "/troy_mcclure"  # Путь к папке для сохранения аугментированных изображений
#
# # # Создайте целевую папку, если её нет
# # os.makedirs(target_dir, exist_ok=True)
# 
# # Определите трансформации
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),      # Случайное отражение по горизонтали
#     transforms.RandomRotation(20),              # Случайный поворот на ±20 градусов
#     transforms.ColorJitter(brightness=0.2,
#                            contrast=0.2,
#                            saturation=0.2,
#                            hue=0.1),           # Изменение яркости, контраста и т.д.
#     # transforms.RandomResizedCrop(size=340),      # Случайная обрезка и изменение размера
# ])
#
# # Применение аугментации ко всем файлам
# for filename in os.listdir(source_dir):
#     # Путь к изображению
#     file_path = os.path.join(source_dir, filename)
#
#     # Открываем изображение
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Только изображения
#         image = Image.open(file_path)
#
#         # Сколько копий вы хотите создать
#         for i in range(50):
#             augmented_image = transform(image)
#
#             # Сохраняем изображение
#             new_filename = f"aug_{i}_{filename}"
#             augmented_image.save(os.path.join(target_dir, new_filename))
#
# print("Аугментация завершена!")

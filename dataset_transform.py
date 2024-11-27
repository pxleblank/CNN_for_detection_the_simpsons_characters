from imports import *
from utilities import CustomImageFolder


def create_transformer():
    # Transformation for train data
    train_transformer = transforms.Compose([
        transforms.Resize((img_row, img_cols)),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    # Transformation for test data
    test_transformer = transforms.Compose([
        transforms.Resize((img_row, img_cols)),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    return train_transformer, test_transformer


def preparing_data():
    train_transformer, test_transformer = create_transformer()
    data_train = datasets.ImageFolder(train_path, transform=train_transformer)
    data_test = CustomImageFolder(root_dir=test_path, transform=test_transformer, class_to_idx=data_train.class_to_idx)
    return data_train, data_test


def dataload(data_train, data_test):
    # Dividing the data (data_train) into training and validation (80/20)
    data_size = len(data_train)
    validation_split = .2
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                             sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=500, shuffle=False)
    return train_loader, val_loader, test_loader


# print(preparing_data()[0].class_to_idx)
# print(f"Number of test images: {len(preparing_data()[1])}")

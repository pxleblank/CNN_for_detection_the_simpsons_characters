from imports import *
from model import ModelVGG11BN, ResNet


def model_create():
    model = ModelVGG11BN(num_classes)  # Create our CNN model.
    # model = ResNet(num_classes)  # Create our CNN model.
    model.to(device)  # Transfer it to device.
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='max',
                                                              factor=0.1,
                                                              patience=10,
                                                              threshold=0.01,
                                                              threshold_mode='rel',
                                                              cooldown=0,
                                                              min_lr=0,
                                                              eps=1e-08
                                                              )

    checkpoint = {
        'state_model': model.state_dict(),
        'state_opt': optimizer.state_dict(),
        'state_lr_scheduler': lr_scheduler.state_dict(),
        'loss': {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': best_loss,
        },
        'metric': {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_precision_per_epoch': val_precision_per_epoch,
            'val_recall_per_epoch': val_recall_per_epoch,
        },
        'lr': lr_list,
        'epochs': epochs
    }

    return model, optimizer, loss_func, lr_scheduler, checkpoint

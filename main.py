from imports import *
from preparing_model import model_create
from dataset_transform import preparing_data, dataload
from model import ModelVGG11BN
from testing import test_model
from training import train_model
from imports import output_model_dir, output_graphics_dir, device
from graphs import show_graphs
from utilities import l1_regularization, l2_regularization
import torch


def main():
    data_train, data_test = preparing_data()

    (train_loader, val_loader, test_loader) = dataload(data_train, data_test)

    model, optimizer, loss_func, lr_scheduler, checkpoint = model_create()

    if __name__ == "__main__":

        print(f"Started training model")
        train_model(model, optimizer, loss_func, lr_scheduler, checkpoint, train_loader, val_loader, data_train)
        print(f"Finished training for whole epochs")

        torch.save(checkpoint, output_model_dir + 'model_state_dict_70_epoch_ResNet.pt')

    print(f"Started testing model")
    test_model(model, loss_func, test_loader, data_test)
    print("Finished testing model")

    show_graphs()


main()

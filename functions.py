from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torchvision.transforms import v2
import random
import matplotlib.pyplot as plt
import os

from constants import RNG

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import shufflenet_v2_x0_5
from torch.optim import lr_scheduler


class ConvNet(nn.Module):
    # A simple implementation of a CNN-based model
    def __init__(
        self,
        channels=20,
        kernel_size=(3, 3),
        imshape=(3, 96, 96),
        dropout=0.1,
    ):
        super().__init__()

        dense_size = 4 * channels

        act = nn.SiLU()
        drop2d = nn.Dropout2d(dropout)
        drop = nn.Dropout(dropout)

        layers = [
            nn.Conv2d(imshape[0], channels, kernel_size),
            nn.Conv2d(channels, channels, kernel_size),
            act,
            drop2d,
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, kernel_size),
            nn.Conv2d(channels, channels, kernel_size),
            act,
            drop2d,
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(dense_size, 30),
            act,
            drop,
            nn.Linear(30, 10),
            act,
            nn.Linear(10, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class TransferNet(nn.Module):
    # Pre-trained ShuffleNetV2 for transfer learning
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.model = nn.Sequential(
            shufflenet_v2_x0_5(weights="IMAGENET1K_V1"),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, X):
        return self.model(X)


def train_model(
    model,
    X_train,
    label_tensor,
    Xvalid_tensor,
    y_valid,
    lr_ival=1000,
    epochs=6000,
    batch_size=1024,
    augmented=False,
    optimizer=None,
    scheduler=None,
):
    # Optimizer and Learning Rate Scheduler
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    if scheduler is None:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, lr_ival, eta_min=0.0001
        )

    # Loss function for binary classification
    lossFN = nn.BCEWithLogitsLoss()

    # Perform training
    model, loss_hist, valid_hist, lr_hist = run_training(
        model,
        optimizer,
        scheduler,
        lossFN,
        X_train,
        label_tensor,
        Xvalid_tensor,
        y_valid,
        epochs=epochs,
        batch_size=batch_size,
        augmented_steps=augmented,
    )

    # Plot history of training loss and performance metrics
    plot_hist(loss_hist, valid_hist, lr_hist)

    # Show confusion matrix based on validation data predictions
    model.eval()
    pred = (nn.functional.sigmoid(model(Xvalid_tensor)) > 0.5).tolist()
    cm = confusion_matrix(y_valid, pred)
    cplt = ConfusionMatrixDisplay(cm)
    cplt.plot()
    plt.show()


def plot_hist(
    loss_hist,
    valid_hist,
    lr_hist,
    av_delay=50,
    folder=None,
    name=None,
    show=True,
    save=False,
):
    # Plot the loss and performance metrics from training the model, per training step

    plt.plot(loss_hist, "c.")
    plt.gcf().set_size_inches(8, 6)
    ax = plt.gca()

    X, F1, ACC = list(zip(*valid_hist))
    ax.plot(X, F1, "g.")
    ax.plot(X, ACC, ".", color="tab:orange")

    if len(loss_hist) > av_delay:
        # Optionally, delay plotting moving averages until things stabilize
        ax.plot(
            np.arange(av_delay, len(loss_hist)), MAV(loss_hist[av_delay:], 50), "b--"
        )
        idx = -1
        for i in range(len(X)):
            if X[i] > av_delay:
                idx = i
                break

        if idx >= 0:
            ax.plot(X[idx:], MAV(F1[idx:], 50), "k--")
            ax.plot(X[idx:], MAV(ACC[idx:], 50), "r--")

    ax.legend(["BCELoss", "F1Score", "Accuracy"])
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(X, lr_hist, "m.-", alpha=0.7)
    ax2.set_ylabel("Learning Rate")

    plt.tight_layout()

    if save:
        assert (
            folder is not None and name is not None
        ), "Must specify folder and file name to save."
        plt.savefig(os.path.join(folder, name + ".png"))

    if show:
        plt.show()
    else:
        plt.close()


def plot_brightness(brightness, b_std, train_labels, std_thr_L, std_thr_H):
    # Plot brightness statistics for all images in the dataset
    fig, ax = plt.subplots(2, 2)
    axs = fig.axes
    fig.set_size_inches(12, 8)

    # Brightness histograms
    tmp = brightness * train_labels
    tmp2 = brightness * (b_std < std_thr_L)
    tmp3 = brightness * (b_std > std_thr_H)
    axs[0].hist(
        [brightness, tmp[tmp > 0], tmp2[tmp2 > 0], tmp3[tmp3 > 0]], 30, histtype="step"
    )
    axs[0].legend(
        [f"StdDev>{std_thr_H}", f"StdDev<{std_thr_L}", "Tumor Present", "Brightness"]
    )
    axs[0].set_xlabel("Brightness")
    axs[0].set_ylabel("Count")

    # Log-scale brightness histograms
    axs[2].hist(
        [brightness, tmp[tmp > 0], tmp2[tmp2 > 0], tmp3[tmp3 > 0]],
        30,
        histtype="step",
        log=True,
    )
    axs[2].set_xlabel("Brightness")
    axs[2].set_ylabel("Count")

    # St Dev of brightness histograms
    tmp4 = b_std * train_labels
    axs[1].hist([b_std, tmp4[tmp4 > 0]], 50, histtype="step")
    axs[1].vlines([std_thr_L, std_thr_H], 0, 30000, linestyles="dashed", color="k")
    axs[1].legend(["Tumor Present", "StdDev"])
    axs[1].set_xlabel("StdDev of Brightness")
    axs[1].set_ylabel("Count")

    # Log-scale St Dev of brightness histograms
    axs[3].hist([b_std, tmp4[tmp4 > 0]], 50, histtype="step", log=True)
    axs[3].vlines([std_thr_L, std_thr_H], 0, 30000, linestyles="dashed", color="k")
    axs[3].set_xlabel("StdDev of Brightness")
    axs[3].set_ylabel("Count")

    plt.show()


def plot_images(
    idxs, train_data, train_labels, lim=100, C=10, S=1.3, highlights=True, title=None
):
    # For plotting example images from the numpy array

    labels = train_labels[idxs]
    # Matplotlib expects a different feature order than pytorch
    tmp = np.reshape(train_data[idxs], (-1, 96, 96, 3))
    print(len(tmp))

    # Number of rows to fit all images given the number of columns
    R = min(int(np.ceil(len(tmp) / C)), int(np.ceil(lim / C)))

    fig, ax = plt.subplots(R, C)
    axs = fig.axes
    fig.set_size_inches(S * C, S * R)

    # Plot each image, highlighting those with cancerous cells present
    for i in range(min(R * C, len(tmp))):
        axs[i].imshow(tmp[i])
        if highlights:
            if labels[i]:
                axs[i].patch.set_edgecolor("r")
                axs[i].patch.set_linewidth(10)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    for j in range(i + 1, R * C):
        axs[j].axis("off")

    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def random_batch(data, labels, batch_size=512):
    # Get a randomly chosen batch (without replacement) from a
    # numpy array, tensor, or dataframe
    idxs = RNG.choice(len(data), batch_size, replace=False)
    return data[idxs], labels[idxs]


def MAV(values, L=10):
    # Moving average
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + (v - out[-1]) / L)
    return out


def apply_random_augmentation(data, p=0.5):
    # Apply any of five random image augmentations (to a tensor) with probability "p" each
    tform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p),
            v2.RandomVerticalFlip(p),
        ]
    )
    trot = v2.RandomRotation(degrees=(0, 10))
    tcrop = v2.RandomResizedCrop((96, 96), scale=(0.9, 1))
    tcol = v2.ColorJitter(0.3, 0.3, 0.3)

    tmp = tform(data)
    if random.random() > p:
        tmp = trot(tmp)
    if random.random() > p:
        tmp = tcrop(tmp)
    if random.random() > p:
        tmp = tcol(tmp)
    return tmp


def run_training(
    model,
    opt,
    sched,
    lossFN,
    X_train,
    label_tensor,
    Xvalid_tensor,
    y_valid,
    epochs=6000,
    batch_size=1024,
    augmented_steps=True,
):
    loss_hist = []
    valid_hist = []
    lr_hist = []

    # Run model training
    best_params = None
    best_acc = 0
    best_f1 = 0
    for i in tqdm(range(epochs // 2 if augmented_steps else epochs)):
        # Select a random batch of images
        batch, targets = random_batch(X_train, label_tensor, batch_size)

        # One training iteration
        opt.zero_grad()
        cu_data = torch.FloatTensor(batch).cuda()
        pred = model(cu_data)
        loss = lossFN(torch.squeeze(pred), targets)
        loss_hist.append(loss.item())
        loss.backward()
        opt.step()
        sched.step()

        if augmented_steps:
            # One augmented iteration
            opt.zero_grad()
            cu_data = apply_random_augmentation(cu_data)
            pred = model(cu_data)
            loss = lossFN(torch.squeeze(pred), targets)
            loss_hist.append(loss.item())
            loss.backward()
            opt.step()
            sched.step()

        if (augmented_steps and i % 5 == 0) or (not augmented_steps and i % 10 == 0):
            # Calculate validation metrics at regular intervals
            lr_hist.append(sched.get_last_lr())
            # Since dropout is being used we must switch the model to evaluation mode (turn off dropout)
            model.eval()
            pred = (nn.functional.sigmoid(model(Xvalid_tensor)) > 0.5).tolist()
            acc = accuracy_score(pred, y_valid)
            f1 = f1_score(pred, y_valid)
            valid_hist.append([2 * i if augmented_steps else i, f1, acc])
            # Remember to switch back to training mode
            model.train()

            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_params = model.state_dict()

    # Load the best parameters into the model
    model.load_state_dict(best_params)

    print(f"Best Accuracy: {best_acc:.2f}")
    print(f"Best F1 Score: {best_f1:.2f}")

    return model, loss_hist, valid_hist, lr_hist

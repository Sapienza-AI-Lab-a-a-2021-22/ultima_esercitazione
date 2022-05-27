import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, random_split
from torchvision import datasets, transforms
from torchinfo import summary
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import wandb
wandb.login(key="d9b4967cc0319eb117ec99a2ac1052f732f96b7d")
wandb.init(project="AI-Lab")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

# Control parameters
do_initial_training = True
random_seed = 41
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


class MLP(nn.Module):
    def __init__(self, input_size=784, output_size=10, layers=[500, 128]):
        super().__init__()
        # construct the layers for MLP
        self.inout_list = [input_size] + layers + [output_size]
        self.layers = nn.Sequential()
        for i in range(len(self.inout_list) - 1):
            self.layers.append(nn.Linear(self.inout_list[i], self.inout_list[i + 1]))

    def forward(self, x):
        # construct the forward propagation mechanism here
        for i, fc in enumerate(self.layers):
            x = F.relu(fc(x))
        return F.softmax(x, dim=1)


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()
    print(ps)
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                             'Trouser',
                             'Pullover',
                             'Dress',
                             'Coat',
                             'Sandal',
                             'Shirt',
                             'Sneaker',
                             'Bag',
                             'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


def train(epoch, model, optimizer, loss_function, train_loader, scheduler=None, log=False):
    # Set model to training mode
    model.train()
    # Loop over each batch from the training set
    correct = 0
    epoch_loss = 0
    accuracy = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        targets = targets.to(device)
        # Pass data through the network
        preds = model(data.view(data.shape[0], -1))
        # Calculate loss
        loss = loss_function(preds, targets)
        epoch_loss += loss.item() * len(data)
        correct += torch.sum(torch.argmax(preds, dim=1) == targets)
        # Zero gradient buffers
        optimizer.zero_grad()
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()
        # eventually, step the scheduler
        if scheduler:
            scheduler.step()
    accuracy = 100. * correct / len(train_loader.sampler)
    epoch_loss /= len(train_loader.sampler)
    if log:
        wandb.log({"Training loss":epoch_loss})
        wandb.log({"Training accuracy":accuracy})
    print(f'Train Epoch: {epoch} Loss: {epoch_loss:.6f}\tAccuracy: {accuracy:.0f}%')
    return epoch_loss, accuracy


def validate(model, validation_loader, test=False):
    model.eval()
    val_loss, correct = 0, 0
    for data, targets in validation_loader:
        data = data.cuda()
        targets = targets.cuda()
        preds = model(data.view(data.shape[0], -1))
        val_loss += loss_function(preds, targets).item() * len(data)
        correct += torch.sum(torch.argmax(preds, dim=1) == targets)
    val_loss /= len(validation_loader.sampler)
    accuracy = 100. * correct / len(validation_loader.sampler)
    if test:
        out_string = "Test"
    else:
        out_string = "Validation"
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        out_string, val_loss, correct, len(validation_loader.sampler), accuracy))
    return val_loss, accuracy


if __name__ == "__main__":
    #FIXME questo qui sotto non credo serva in questo caso
    if not os.path.exists("../data"):
        dataset = datasets.MNIST(root="../data", train=True, transform=transforms.ToTensor, download=True)
    else:
        dataset = datasets.MNIST(root="../data", transform=transforms.ToTensor(), train=True)

    # Define a transform to normalize the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])

    # Download and load training and test data
    trainset, validationset = random_split(
        datasets.MNIST(root="data", train=True, transform=transform),
        [1000, 59000])
    testset = datasets.MNIST(root="data", train=False, transform=transform, download=True)

    # If this is the first run of the code, learn some initial weights
    if do_initial_training:
        num_epochs = 200
        train_loader = DataLoader(trainset, batch_size=8196, shuffle=True, pin_memory=True)  # create a data loader here
        print(f"Training set len: {len(trainset)}")
        validation_loader = DataLoader(validationset, batch_size=8196, pin_memory=True)
        print(f"Validation set len: {len(validationset)}")

        model = MLP()
        if os.path.exists(f"initial_weights_{random_seed}.tar"):
            model.load_state_dict(torch.load(f"initial_weights_{random_seed}.tar"))
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.5, steps_per_epoch=len(
            train_loader), epochs=num_epochs)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        loss_function = nn.CrossEntropyLoss()
        for e in range(num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"lr": current_lr})
            train(e, model, optimizer, loss_function, train_loader, scheduler, log=True)
            if e % 10 == 0:
                val_loss, acc = validate(model, validation_loader)
                wandb.log({"Initial val loss":val_loss})
                wandb.log({"Initial accuracy":acc})
            # scheduler.step(acc)
        torch.save(model.state_dict(), f"initial_weights_{random_seed}.tar")

    # CROSS VALIDATION EXAMPLE
    """
    Remember: in designing DNN we usually follow the practical rule of using the most complex network
    our system can handle, while avoiding overfitting by increasing the regularization. This can be 
    done in different ways, but the most simple is weight-decay (a.k.a. L2/L1 regularization). In this
    case, we still have to find the best balance between the loss we are minimizing and the regularization
    term that limits the flexibility of our network. We will call "alpha" the factor that express this 
    balace and use K-Fold cross validation to find the best value. This will be possible because the 
    network we are using is small. In real cases one has to plan well the experiments, since each 
    training run will take hours, days, or more.
    """
    # new imports
    from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, ParameterGrid
    #load initial model
    model = MLP()
    model.to(device)
    summary(model,input_size=(1,1,784))
    initial_state_dict = torch.load(f"initial_weights_{random_seed}.tar")
    test_loader = DataLoader(testset, batch_size=8192)
    # hyperparameter: weight decay factor. The list of values are separated with an exponential pace
    alpha_list = list(np.logspace(-5, 4, num=10, base=2))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    epochs = 20
    loss_function = nn.CrossEntropyLoss()

    best_model = None
    best_param = None
    best_validation_loss = 1e9 #something big
    loss_vector = list()
    for alpha in alpha_list:
        print(f'================================')
        print(f"    Using alpha = {alpha}")
        print(f'================================')
        for fold, (train_ids, val_ids) in enumerate(kf.split(trainset)):
            print(f'FOLD {fold}')
            print('--------------------------------')
            # init training loop
            model.load_state_dict(initial_state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=alpha)
            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(val_ids)
            train_loader = DataLoader(trainset, batch_size=4096, sampler=train_subsampler, pin_memory=True)
            valid_loader = DataLoader(trainset, batch_size=8192, sampler=valid_subsampler)
            # perform training
            for epoch in range(epochs):
                fold_epoch_loss, fold_tr_accuracy = train(epoch, model, optimizer,
                                                       loss_function, train_loader)
                wandb.log({f"fold {fold} training loss": fold_epoch_loss})
                wandb.log({f"fold {fold} accuracy": fold_tr_accuracy})

            # compute validation scores
            fold_loss, fold_accuracy = validate(model, valid_loader)
            loss_vector.append(fold_loss)
            # save model
            if not os.path.exists("weights"):
                os.mkdir("weights")
            torch.save(model.state_dict(), f"weights/valid_model_{fold:02d}.tar")

        # average validation loss among the folds
        validation_loss = np.mean(loss_vector)
        wandb.log({f"validation_loss_{alpha:2.6f}": validation_loss})
        print(f"Validation loss for fold {alpha:2.6f}: {validation_loss}")
        # check best model and parameter
        if validation_loss <= best_validation_loss:
            best_model = model
            best_validation_loss = validation_loss
            best_param = alpha

    print(f'End cross validation')
    print('================================')
    wandb.log({"best_validation_loss":best_validation_loss})

    # test the best model on the test set
    final_loss, final_accuracy = validate(best_model, test_loader)
    wandb.log({"final_loss": final_loss, "final_accuracy": final_accuracy})
    print(f"final loss: {final_loss}\nfinal accuracy: {final_accuracy}")






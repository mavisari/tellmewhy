# custom modules
from data_utils import MRIDataset
# general libraries
from pathlib import Path
import pickle
import copy
from tqdm import tqdm
import numpy as np
import random
import cv2
#import umap
import json
import math
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
# evaluation
import torchmetrics
import torchmetrics.classification
from torchmetrics import ConfusionMatrix, AUROC
from mlxtend.plotting import plot_confusion_matrix
# plotting
import PIL
from PIL import Image
import matplotlib.pyplot as plt
# huggingface
from datasets import load_dataset
from huggingface_hub import login
# ALWAYS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#####################################   DATA   #############################################
# loading dataset from huggingface
def load_dataset_from_huggingface():
    """Load the dataset, the name is 'Simezu/brain-tumour-MRI-scan' and can be downloaded directly from huggingface hub using the dataset library.
    Args:
        None
    Returns:
        dataset (dict): the dataset with PIL images
    """
    my_token = os.getenv("HF_TOKEN")
    dataset = load_dataset('Simezu/brain-tumour-MRI-scan', token=my_token)
    dataset.save_to_disk('data')
    return dataset#.with_format('torch')


def extract_labels_from_json(json_file='data/train/dataset_info.json'):
    """Return the mapping of class names and encoded labels:
        {'1-notumor': 0, '2-glioma': 1, '3-meningioma': 2, '4-pituitary': 3}
    """
    data = []
    with open(Path(json_file), 'r') as f:
        data = json.load(f)
    
    raw_info = [x.split('/') for x in list(data['download_checksums'].keys())]
    classes = list(set([x[-2] for x in raw_info]))
    class_to_idx = {x:int(x[0])-1 for x in sorted(classes)}
    return class_to_idx


def preprocess_datasets(dataset):
    """From loaded images to images converted to RGB channels. This function is needed because the images have different modes (L, P, RGBA, RGB), and we want just RGB.
    Args:
        dataset (dict): a dictionary as extracted from huggingface
    Returns:
        datasets, labels
    """
    train_dataset, test_dataset = dataset['train']['image'], dataset['test']['image']
    train_label, test_label = dataset['train']['label'], dataset['test']['label']
    
    def normalize_channels(x):
        x_ = x
        if x.mode in ['L', 'P', 'RGBA']:
            x_ = x.convert('RGB')
        return x_
    
    def pil_transform():
        transf = transforms.Compose([transforms.Resize((224,224))])
        return transf
    
    pill_transform = pil_transform()
    
    for i,x in enumerate(train_dataset):
        x  = normalize_channels(x)
        #train_dataset[i] = pill_transform(x)
        train_dataset[i] = x.resize((224,224))
    
    for i,x in enumerate(test_dataset):
        x = normalize_channels(x)
        #train_dataset[i] = pill_transform(x)
        test_dataset[i] = x.resize((224,224))
    
    return train_dataset, test_dataset, train_label, test_label


def get_dataset_and_loader(dataset, 
                    label,
                    split,
                    BATCH_SIZE=128):
    """Images contained in the brain dataset have different dimensions, so we need to resize them to 224x224
    Args:
        dataset (torch.nn.dataset): the brain dataset as loaded using the func above
        RESIZE (int): the target image dimension (i.e. 224, for 224x224 shape)
        BATCH_SIZE (int): the batch size for dataloaders default 128
    Returns:
        train_dataloader
        test_dataloader
    """
    shuffle = True
    if split == 'test':
        shuffle = False

    data_transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize(mean=[.5], std=[.5]) # this will result into very WRONG images
        ])
    
    dataset_ = MRIDataset(data=dataset, 
                            targets=label, 
                            split=split, 
                            transform=data_transform)

    loader = DataLoader(dataset=dataset_, batch_size=BATCH_SIZE, shuffle=shuffle)
    #train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    #train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=shuffle)
    #test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=shuffle)
    
    return dataset_, loader
    
    
#####################################   MODELS   #############################################
# saving model weights to file
def save_model_to_file(name, model, save_dir):
    """Saves a model's weights to file.
    Args:
        name (str): name of the model
        model (torch.nn): the model from which to extract the weights
        save_dir (str): the directory in which you want to save the model, if it does not exist, it will be automatically created.
    """
    # create model directory path
    MODEL_PATH = Path(save_dir)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    # model save path
    MODEL_NAME = name if name.endswith('.pth') else name+'.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # save the model dict
    torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)
    
    
# loading model weights from file
def load_model_from_file(name, model, load_dir):
    """Loads a model's weights from file.
    Args:
        name (str): name of the model
        model (torch.nn): an instance of the 'name' model
        load_dir (str): the directory in which the model weights are stored.
    Returns:
        (torch.nn)
    """
    MODEL_PATH = Path(load_dir)
    MODEL_NAME = name if name.endswith('.pth') else name+'.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    return model


# save model's training history to file
def save_model_history_pkl(model:torch.nn.Module,
                            history: list,
                            save_dir: str):
    MODEL_PATH = Path(save_dir)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    # history save path
    name = model.__class__.__name__
    MODEL_NAME = name+'_history_pickle'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    pkl_file = open(MODEL_SAVE_PATH, 'wb')
    pickle.dump(history, pkl_file)
    pkl_file.close()
    
# load model's training history from file
def load_model_history_pkl(model: torch.nn.Module, load_dir: str):
    MODEL_PATH = Path(load_dir)
    name = model.__class__.__name__
    MODEL_NAME = name+'_history_pickle'
    MODEL_LOAD_PATH = MODEL_PATH / MODEL_NAME
    pkl_file = open(MODEL_LOAD_PATH, 'rb')
    history = pickle.load(pkl_file)
    pkl_file.close()
    return history
    
    
#####################################   TRAINING UTILS   #############################################
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.
    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        (torch.float): Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# train function
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader, # i.e., train_loader
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: accuracy_fn,
               device: torch.device=device):
    """Performs training step with model dataloader, for a single epoch, return a history dictionary tracking the training progress
    Args:
        model (torch.nn.Module): instanced model
        data_loader (torch.utils.data.DataLoader): train dataloader
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optmizer
        accuracy_fn (function): accuracy function
        device (torch.device): target device where we perform the calculations (cpu or CUDA, if available)
    """
    train_loss, train_acc = 0,0
    
    model = model.to(device)
    model.train()
    
    for X,y in tqdm(data_loader):
        optimizer.zero_grad()
        # put data in target device
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        
        y = y.squeeze().long()
        loss = criterion(y_pred, y)
        
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss.backward()
        optimizer.step()
    
    # divide train loss and accuracy per batch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    history = {'model': model.__class__.__name__,
               'train_loss': train_loss,
               'train_accuracy': train_acc}
    print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%')
    return history
    

# evaluation (for eval and test dataloaders)
def test(model: torch.nn.Module,
         split,
         data_loader: torch.utils.data.DataLoader, # i.e., train_loader
         criterion: torch.nn.Module,
         accuracy_fn: accuracy_fn,
         device: torch.device=device):
    """Performs trevaluation aining step with model dataloader, for a single epoch
    Args:
        model (torch.nn.Module): instanced model
        split (str): internal for medMNIST, either 'train' or 'test'
        data_loader (torch.utils.data.DataLoader): dataloader
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optmizer
        accuracy_fn (function): accuracy function
        device (torch.device): target device where we perform the calculations (cpu or CUDA, if available)
    Returns:
        y_score (torch tensor): the predicted scores, to get the labels, use torch.softmax(y_score.squeeze(), dim=0).argmax(dim=1)
        history: dictionary with the model name, test loss and test accuracy
    """
    # metrics
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    test_loss, test_acc = 0,0
    
    model.eval()
    with torch.inference_mode():
        for X,y in tqdm(data_loader, desc='make predictions...'):
            X, y = X.to(device), y.to(device)
            # forward pass
            test_pred = model(X)
            
            y = y.squeeze().long()
            test_loss += criterion(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            y = y.float().resize_(len(y), 1)
            
            # calculate scores
            y_true = torch.cat((y_true, y), 0)
            y_score = torch.cat((y_score, test_pred), 0)
            
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        y_true = y_true.cpu()
        y_score = y_score.detach().cpu()
    
    # save history dictionary 
    history = {'model': model.__class__.__name__,
                'test_loss': test_loss,
                'test_accuracy': test_acc}
    print(f'{split} loss: {test_loss:.5f} | {split} acc: {test_acc:.2f}%')
    return y_score, history



# full training loop:
def fit(NUM_EPOCHS: int,
        model: torch.nn.Module,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        split: str,
        criterion: torch.nn.Module,
        optimizer: torch.nn.Module,
        accuracy_fn=accuracy_fn,
        device=torch.device):
    """Fit the model on training data.
    Args:
        n_epochs (int): number of epochs to train the model
        model (torch.nn.Module): instanced model
        train_data_loader (torch.utils.data.DataLoader): train dataloader
        test_data_loader (torch.utils.data.DataLoader): test dataloader
        split (int): split for test evaluation
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optmizer
        accuracy_fn (function): accuracy function
        device (torch.device): target device where we perform the calculations (cpu or CUDA, if available)
    Returns:
        full_history (list of tuples): the list contains the results as dicts of train and test as a tuple (train, test) for each epoch
    """
    full_history = []
    for epoch in range(NUM_EPOCHS):
        print(f'epoch {epoch+1}\n---------------')
        full_history.append((
            # train step
            train_step(model=model,
                   data_loader=train_data_loader,
                   criterion=criterion,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device),
            # right now: validation on test set (todo: perform on validation set)
            test(model=model,
                   data_loader=test_data_loader,
                   split=split,
                   criterion=criterion,
                   accuracy_fn=accuracy_fn,
                   device=device))
        ) 
           
    return full_history

def fit_early_stopping(NUM_EPOCHS: int,
        model: torch.nn.Module,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        split: str,
        criterion: torch.nn.Module,
        optimizer: torch.nn.Module,
        patience: int,
        accuracy_fn=accuracy_fn,
        device=torch.device):
    """Fit the model on training data.
    Args:
        n_epochs (int): number of epochs to train the model
        model (torch.nn.Module): instanced model
        train_data_loader (torch.utils.data.DataLoader): train dataloader
        test_data_loader (torch.utils.data.DataLoader): test dataloader
        split (int): split for test evaluation
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optmizer
        accuracy_fn (function): accuracy function
        device (torch.device): target device where we perform the calculations (cpu or CUDA, if available)
        patience (int): Number of events to wait if no improvement and then stop the training.
    Returns:
        full_history (list of tuples): the list contains the results as dicts of train and test as a tuple (train, test) for each epoch
    """
    full_history = []
    best_loss = float('inf')
    best_model_weights = None
    best_epoch_index = 0
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        print(f'epoch {epoch+1}\n---------------')
        train_res = train_step(model=model,
                   data_loader=train_data_loader,
                   criterion=criterion,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device)   
        test_res = test(model=model,
                   data_loader=test_data_loader,
                   split=split,
                   criterion=criterion,
                   accuracy_fn=accuracy_fn,
                   device=device)
        full_history.append((train_res, test_res))
        current_test_loss = test_res[1]['test_loss'].item()

        if current_test_loss < best_loss:
            best_loss = current_test_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch_index = epoch
            print(f"Best loss: {best_loss:.4f}. Save model weights.")
        else:
            epochs_no_improve += 1
            print(f"No improvement: {epochs_no_improve}/{patience}")
        if epochs_no_improve == patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    print(f"Loading best model weights from epoch {best_epoch_index+1} (Loss: {best_loss:.4f})")
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    return full_history

# predict on a sample
def predict(model:torch.nn.Module,
            samples: list, # obtained from list(dataset)
            device: torch.device):
    """Test the model on sample data.
    Args:
        model (torch.nn.Module): instanced model
        sample_loader (list): sample taken from a torch dataset
        device (torch.device): target device where we perform the calculations (cpu or CUDA, if available)
    Return:
        A torch tensor with the predictions, logits, for each class.
    """
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
            pred_probs.append(pred_prob)
    return torch.stack(pred_probs)


# feature extraction
def extract_features(model:torch.nn.Module,
                     data_set:torch.utils.data.Dataset,
                     device=device):
    """
    REWRITE!!!!
    
    Args:
        model (torch nn module): the trained model
        data_set (torch dataset): the dataset, list of X,y tuples for which to extract the features
        device: the device used to run the model
    Returns:
        tuple containing:
        features (np.array), labels (list of strings)
    """
    #model.fc = torch.nn.Identity()
    model_encoder = '' # replace any model with model_encoder in this func
    
    feature_test = []
    feature_labels = []
    with torch.no_grad():
        model = model.to(device)
        for sample, label in tqdm(data_set, desc='Extracting features...'):
            embedding = model(sample.unsqueeze(dim=0).to(device))
            feature_test.append(embedding.cpu())
            feature_labels.append(label)
            
    feature_test = np.vstack(feature_test)
    return feature_test, feature_labels


#####################################   VISUALIZATION   #############################################
# plot a random image from the dataset
def plot_rand_image(data):
    img = random.choice(data)[0]
    img = img.permute(1,2,0)
    plt.axis('off')
    plt.imshow(img.cpu())

# plot predictions
def plot_predictions(test_samples,
                     test_labels, 
                     predictions, 
                     model,
                     class_names):
    """Plot the predictions in a grid of nxn samples with true and predicted labels.
    If the labels do not match, plot them in red, otherwise in green.
    
    Args:
        test_samples (list): list of samples for which the prediction was made.
        test_labels (list): list of labels for the test samples
        predictions (list): predicted labels for the sample
        model (torch.nn.Model): an instantiated model
        class_names (dict): dictionary containing the mappings between class names and codes, e.g. 0: Adipose, 1: Stroma, etc.
    Return:
        optional todo, Figure
    """
    pred_classes = predictions.argmax(dim=1)
    # plot predictions
    plt.figure(figsize=(12,14))
    # prepare the grid for n samples
    nr = int(math.sqrt(len(test_samples)))
    nc = nr
    for i, sample in enumerate(test_samples):
        plt.subplot(nr, nc, i+1)
        sample = sample.permute(1,2,0) # for c,h,w images
        sample = sample.cpu()
        plt.axis('off')
        plt.imshow(sample)
        pred_label = class_names[str(pred_classes[i].item())]
        #truth_label = class_names[str(test_labels[i][0])] for pathmnist
        truth_label = class_names[str(test_labels[i])]
        title_text = f'Pred: {pred_label} \n Truth: {truth_label}'
        if pred_label == truth_label:
            plt.title(title_text, fontsize=14, c='g')
        else:
            plt.title(title_text, fontsize=14, c='r')
            

#####################################   PERFORMANCE EVALUATION   #############################################
def confusion_matrix(model, 
                     y_score,
                     data_set, 
                     class_names, 
                     normalize=False,
                     plot=True):
    """
    Args:
        model (torch.nn.Module): model
        y_score (torch.tensor): tensor containing the predicted logits for the test dataset
        data_set (dataset): the test dataset
        class_names (dict): the dict of class names
        normalize (bool): if true, plots the percentages instead of numbers
        plot (bool): a flag to check if we want to plot the confusion matrix, default yes
    Returns:
        the confusion matrix (and plots)
    """
    # note the distribution of classes in testset (brain mri): [no: 405, g: 300, m: 306, p: 300]
    
    y_preds_labels = torch.softmax(y_score.squeeze(), dim=1).argmax(dim=1)
    true_labels = torch.tensor(data_set.labels).squeeze()
    confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
    confmat_numpy = confmat(preds=y_preds_labels,
                             target=true_labels).numpy()
    
    if normalize:
        confmat_numpy = (confmat_numpy*100)/confmat_numpy.sum(axis=1)
    
    if plot:
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat_numpy,
            class_names=class_names.values(), 
            figsize=(5,7)
        )
        fig.patch.set_facecolor('whitesmoke')
        ax.set_facecolor('whitesmoke')
        ax.set_title(f'Confusion matrix for {model.__class__.__name__} model')
        ax.title.set_size(16)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        for text in ax.texts:
            text.set_fontsize(14) 

    return confmat_numpy


# AUROC
def AUROC_(model, 
            y_score,
            data_set, 
            class_names):
    """Plot the ROC curve and calculates the area. It takes the samples and the predictions.
       Currently part of plot_curves().
    Args:
        data_set (list): usually it's the test dataset.
        y_score (list): predicted logits
        model (torch.nn.Model): an instantiated model
        class_names (dict): dictionary containing the mappings between class names and codes, e.g. 0: Adipose, 1: Stroma, etc.
    Return:
        None
    """
    #y_preds_labels = torch.softmax(y_score.squeeze(), dim=0).argmax(dim=1)
    true_labels = torch.tensor(data_set.labels).squeeze()
    
    auroc = torchmetrics.classification.MulticlassAUROC(num_classes=len(class_names), average=None, thresholds=None)
    area = auroc(y_score, true_labels)
    
    # plot ROC
    roc = torchmetrics.classification.MulticlassROC(num_classes=len(class_names),thresholds=None)
    roc.update(y_score, true_labels)
    fig_, ax_ = roc.plot(score=True)
    # replacing label index with true name
    handles, labels = ax_.get_legend_handles_labels()
    labels = [f'{class_names[l[:1]]} {l[1:]}' for l in labels]
    ax_.legend(handles, labels)
    ax_.set_title(f'{ax_.get_title()} {model.__class__.__name__}')
    
    return area

# curves (acc, loss, auroc)
def plot_curves(model,
                history,
                y_score,
                data_set, 
                class_names):
    """Plot Loss curves, Accuracy curves, and multiclass ROC curves
    Args:
        model (torch.nn.Module): the trained model
        history (list): a list of dictionaries containing the model's progress
        y_score (list): predicted logits
        data_set (list): usually it's the test dataset.
        class_names (dict): dictionary containing the mappings between class names and codes
    Returns:
        None
    """
    train_history=[tr for tr,te in history]
    test_history=[te[1] for tr,te in history]
    true_labels = torch.tensor(data_set.labels).squeeze()
    
    # roc calculations
    auroc = torchmetrics.classification.MulticlassAUROC(num_classes=len(class_names), average=None, thresholds=None)
    area = auroc(y_score, true_labels)
    roc = torchmetrics.classification.MulticlassROC(num_classes=len(class_names),thresholds=None)
    roc.update(y_score, true_labels)
    
    fig = plt.figure(figsize=(7,7))
    gs = fig.add_gridspec(2,2, height_ratios=[1,3])
    fig.patch.set_facecolor('whitesmoke')
    
    # Loss
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_title("Training and Testing Loss", fontsize = 14)
    ax0.set_facecolor('whitesmoke')
    plt.plot([ep['train_loss'].item() for ep in train_history], label="train")
    plt.plot([ep['test_loss'].item() for ep in test_history], label="test")
    #plt.xticks(range(num_epochs), list(range(1,num_epochs+1)))#[1,2,3])
    ax0.legend(loc='upper left')
    
    # Accuracy
    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_facecolor('whitesmoke')
    ax1.set_title('Training and Testing Accuracy', fontsize = 14)
    ax1.set_ylim(0,100)
    plt.plot([ep['train_accuracy'] for ep in train_history], label="train")
    plt.plot([ep['test_accuracy'] for ep in test_history], label="test")
    #plt.xticks(range(num_epochs), list(range(1,num_epochs+1)))#[1,2,3])
    ax1.legend(loc='upper left')
    
    # ROC
    ax2 = fig.add_subplot(gs[1,:])
    ax2.set_title('ROC', fontsize = 14)
    ax2.set_facecolor('whitesmoke')
    roc.plot(score=True, ax=ax2)
    handles, labels = ax2.get_legend_handles_labels()
    labels = [f'{class_names[l[:1]]} {l[1:]}' for l in labels]
    ax2.legend(handles, labels, fontsize="large")
    ax2.set_title(f'{ax2.get_title()} {model.__class__.__name__}')
    
    fig.tight_layout(pad=1.0)
    plt.show()
    
    


##################################### (ignore)  FEATURE EXTRACTION   #############################################
# Feature latent space using UMAP
# to visualize the features we need to extract them
def umap_embed(features, labels):
    """Reduces the feature dimensionality to 2 from the model's features. It is then used for visualizing the feature separation.
    Args:
        features (np.array): features extracted from the model using the function "extract_features"
        labels (list): labels for the features
    Returns:
        embedded features in 2 dimensions using umap
    """
    pass





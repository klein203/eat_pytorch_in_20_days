from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils import data
from torchkeras import summary, Model
from sklearn.metrics import precision_score, accuracy_score
import pandas as pd
import os
import datetime


# plot
def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'valid_'+metric])
    plt.show()

def plot_images(features, mean=0.5, std=0.5, nrows=8, figsize=(2, 2)):
    # images: tensor (B, C, H, W), grid_image: ndarray (C, H, W)
    grid_image = utils.make_grid(features, nrow=nrows).numpy()
    
    # imshow (H, W, C)
    grid_image = grid_image.transpose(1, 2, 0)
    grid_image = mean + grid_image * std

    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# save and load
def save_history(model, file, mode='csv'):
    assert mode == 'csv'
    assert type(model.history) is pd.DataFrame
    model.history.to_csv(file)

def save_weight(model, file):
    weights = dict()
    weights.update({'epoch': model.epoch})
    weights.update({'net': model.state_dict()})
    weights.update({'optimizer': model.optim.state_dict()})
    torch.save(weights, file)

def load_history(file, index_col='epoch', mode='csv'):
    assert mode == 'csv'
    return pd.read_csv(file, index_col=index_col)

def load_weight(model, file, net_only=False):
    weights = torch.load(file)
    model.load_state_dict(weights['net'])
    if not net_only:
        model.epoch = weights.get('epoch', 0)
        model.optim.load_state_dict(weights['optimizer'])
    return model

# metrics
def precision_metrics(targets, labels):
    # targets (-1, C), labels (-1)
    y_pred = targets.data.max(1)[1].numpy()
    y_true = labels.numpy()
    score = precision_score(y_true, y_pred, average='macro')
    # return (1)
    return torch.tensor(score)

def accuracy_metrics(targets, labels):
    # targets (-1, C), labels (-1)
    y_pred = targets.data.max(1)[1].numpy()
    y_true = labels.numpy()
    score = accuracy_score(y_true, y_pred)
    # return (1)
    return torch.tensor(score)

# training functions
def run_step(model, features, labels, train_mode=True):
    targets = model(features)
    
    metrics = dict()
    loss = model.loss_fn(targets, labels)
    metrics.update({'%sloss' % ('' if train_mode else 'val_'): loss.item()})
    
    for metric_name, metric_fn in model.metrics_dict.items():
        metric_value = metric_fn(targets, labels)
        metrics.update({'%s%s' % ('' if train_mode else 'val_', metric_name): metric_value.item()})

    loss.backward()
    model.optim.step()
    model.optim.zero_grad()

    return metrics

def run_epoch(model, dataloader, train_mode=True, log_per_steps=200):
    metrics_epoch = dict()

    model.train(train_mode)
    for step, (features, labels) in enumerate(dataloader, 1):
        metrics = run_step(model, features, labels, train_mode)

        # # update loss_epoch (mean)
        # loss_epoch = (step - 1) / step * loss_epoch + metric_val / step
        # update metric_epoch (mean)
        for metric_name, metric_val in metrics.items():
            if metrics_epoch.get(metric_name) == None:
                metrics_epoch[metric_name] = metric_val
            else:
                metrics_epoch[metric_name] = \
                    (step - 1) / step * metrics_epoch[metric_name] + metric_val / step

        if step % log_per_steps == 0:
            print(" - Step %d, %s" % (step, metrics_epoch))

    return metrics_epoch

def train_model(model, dataloader_train, dataloader_valid, epochs, log_per_epochs=10, log_per_steps=200):
    print("==========" * 6)
    print("= Training model")
    
    metrics_list = []
    start_epoch = 1 + model.epoch
    end_epoch = epochs + 1 + model.epoch
    for epoch in range(start_epoch, end_epoch):
        metrics = dict()
        print("==========" * 6)
        print("= Epoch %d/%d @ %s" % (epoch, end_epoch - 1, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        metrics_train = run_epoch(model, dataloader_train, train_mode=True, log_per_steps=log_per_steps)
        metrics_valid = run_epoch(model, dataloader_valid, train_mode=False, log_per_steps=log_per_steps)
        metrics.update({'epoch': epoch})
        metrics.update(metrics_train)
        metrics.update(metrics_valid)
        metrics_list.append(metrics)

        model.epoch = epoch

        if epoch % log_per_epochs == 0:
            print('= %s' % metrics)
        
    print("==========" * 6)
    
    model.history = pd.DataFrame(metrics_list)
    model.history.set_index('epoch', inplace=True)
    return model.history

def predict_model(model, features):
    model.eval()
    targets = model(features)
    
    return targets.data.max(1)[1]

def eval_model(model, features, labels):
    model.eval()
    targets = model(features)

    metrics = dict()
    for metric_name, metric_fn in model.metrics_dict.items():
        metric_value = metric_fn(targets, labels)
        metrics.update({metric_name: metric_value.item()})
    
    return metrics
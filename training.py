import time
import torch
from confmetrics import f2_score, F1
import copy
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm


def initialize_model(num_classes, model = None, use_pretrained=True, input_size = (224, 224)):
    """
        num_classes - number of classes
        model 
    """
    if model == None:
        model = models.resnet50(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model, input_size


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, train_phases = ['train', 'val']):
    """
        model - initialized model;
        dataloaders - dictionary with dataloaders;
        criterion - training criterion;
        optimizer - training optimizer;
        device - compute device;
        num_epochs - number of epochs to train for;
        train_phases - list of names phases of training;
    """
    since = time.time()

    val_acc_history = []

    best_model_wts = model.state_dict()
    best_acc = 0.0

    print(f'Start training for {num_epochs} epochs')
    for epoch in range(num_epochs):
        for phase in train_phases:
            is_training = phase == 'train'
            if is_training:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            f2 = 0
            f1 = 0

            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), leave=False)
            batch_count = len(dataloaders[phase].dataset)

            for i, (inputs, labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                f2 += f2_score(preds, labels.data)
                f1 += F1(preds, labels.data)
                batch_accuracy = torch.sum(preds==labels.data)
                running_corrects += batch_accuracy
                pbar.set_description(f"Phase {phase} | Loss {loss.item():.5f} | Acc {float(batch_accuracy) / len(labels.data):.4f} | f-1 {f1}")
            
            epoch_loss = running_loss / batch_count
            epoch_acc = running_corrects.double() / batch_count
            epoch_f2 = f2 / batch_count
            epoch_f1 = f1 / batch_count
            pbar.close()
            #print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F-beta score: {epoch_f2:.4f}, F-1 score: {epoch_f1:.4f}')

            if not is_training:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print(f'Best acc now: [{best_acc}]') 
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_history.append(epoch_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}] | Loss ({phase}): {epoch_loss:.4f} '+\
            f' | Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


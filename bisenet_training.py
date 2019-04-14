import torch
import torch.nn.functional as F
from tqdm import tqdm
from segmetation_metrics import IoU
import numpy as np
import copy


debug = False

def prepare_masks(masks, scale_factor = None, with_aux=True):
    """ Make binary mask for background and object, then interpolate to smaller dimensions for auxillary losses. 
    
    Args:
        masks (torch.Tensor): gt masks, batch_size x H x W.
        with_aux (bool, optional): flag to return auxillary masks.
        
    Returns:
        masks (torch.Tensor): batch_size x 2 x H x W mask for background and object.
        masks_16 (torch.Tensor, optional): interpolated 'masks' for auxillary loss.
        masks_32 (torch.Tensor, optional): interpolated 'masks' for auxillary loss.
    """
    if scale_factor == None:
        scale_factor = (0.125, 0.125)
    
    if with_aux:
        masks_16 = scale_to_output(masks, scale_factor)
        masks_32 = scale_to_output(masks, scale_factor)
        masks = scale_to_output(masks, scale_factor)
        
        masks = torch.cat((1-masks, masks), dim=1)
        masks_16 = torch.cat((1-masks_16, masks_16), dim=1)
        masks_32 = torch.cat((1-masks_32, masks_32), dim=1)

        return (masks, masks_16, masks_32)
    else:
        masks = torch.cat((1-masks, masks), dim=1)
        
        return masks
    
 
def scale_to_output(ground_truth, scale_factor = (0.125, 0.125)):
    """
    scale tensor to match WxH dims
    """
   
    ground_truth = F.interpolate(ground_truth, scale_factor=scale_factor, mode='nearest')
    if debug:
        print('ground_truth shape')
        print(ground_truth.shape)
    return ground_truth


def calculate_segmentation_accuracy(pred, mask, metric):
    w = pred.shape[2] / mask.shape[2]
    h = pred.shape[3] / mask.shape[3]
    if debug:
        print('W x H')
        print(w)
        print(h)
    mask = scale_to_output(mask, (w, h))
    metric.add(pred, mask)
    iou, mean_iou = metric.value()
    
    if debug:
        print(iou)
        print(mean_iou)
    
    return iou

def train_step(model, optimizer, train_loader, criterions, device, metric):
    """ One epoch of training. 
    
    Args:
        model: instance of BiSeNet class.
        optimizer: torch.optim.Adam.
        train_loader (torch.DataLoader): dataloader for training data.
        criterions (list): list of criterions (main and auxillary losses).
        device (torch.device): cuda if avaliable.
        
    Returns:
        mean_loss (float): mean loss for an epoch.
        mean_acc (float): mean accuracy for an epoch.
    """
    model.train()
    
    mean_loss = 0
    mean_acc = 0
    total = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (images, masks) in pbar:
        # Place on gpu
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device)
        
        # Forward pass
        outputs, side_out_16, side_out_32 = model(images)
        if debug:
            print('outputs')
            print(outputs.shape)
            print('out_16')
            print(side_out_16.shape)
            print('out_32')
            print(side_out_32.shape)
        # Loss calculation
        w = outputs.shape[2] / masks.shape[2]
        h = outputs.shape[3] / masks.shape[3]
        if debug:
            print('W x H')
            print(w)
            print(h)
        loss_masks = prepare_masks(masks, (w, h), with_aux=True)
        loss = criterions[0](outputs, loss_masks[0])
        loss += criterions[1](side_out_16, loss_masks[1])
        loss += criterions[2](side_out_32, loss_masks[2])
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Some statistics
        mean_loss += loss.item()
        mean_acc += np.nanmean(get_acc(outputs, masks, metric))
        #print(np.nanmean(mean_acc))
        total += masks.size(0)
        pbar.set_description(f"Loss {mean_loss/total:.5f} | Acc {mean_acc/total:.4f} |")
    
    pbar.close()
    return (mean_loss / total, mean_acc / total)


def eval_step(model, eval_loader, criterions, device, metric):
    """ Evaluation of the model. 
    
    Args:
        model: instance of BiSeNet class.
        eval_loader (torch.DataLoader): dataloader for evaluation data.
        criterions (list): list of criterions (main and auxillary losses).
        device (torch.device): cuda if avaliable.
        
    Returns:
        mean_loss (float): mean loss for an evaluation data.
        mean_acc (float): mean accuracy for an evaluation data.
    """
    model.eval()
 
    with torch.no_grad():
        mean_loss = 0
        mean_acc = 0
        total = 0
        for i, (images, masks) in enumerate(eval_loader):
            # Place on gpu
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device)

            # Forward pass
            outputs, side_out_16, side_out_32 = model(images)
            
            w = outputs.shape[2] / masks.shape[2]
            h = outputs.shape[3] / masks.shape[3]
            if debug:
                print('W x H')
                print(w)
                print(h)
            
            loss_masks = prepare_masks(masks, (w, h), with_aux=True)

            loss = criterions[0](outputs, loss_masks[0])
            loss += criterions[1](side_out_16, loss_masks[1])
            loss += criterions[2](side_out_32, loss_masks[2])
            
            # Some statistics
            mean_loss += loss.item()
            mean_acc += np.nanmean(get_acc(outputs, masks, metric))
            #print(mean_acc)
            total += masks.size(0)
            
    return (mean_loss / total, mean_acc / total)


def get_acc(pred, mask, metric):
    return calculate_segmentation_accuracy(pred, mask, metric)


def train(model, opt, train_loader, eval_loader, criterions, n_epochs, device, logdir, start_from_current_accuracy = True):
    """
    Main training loop. Need to calculate model accu
    """
    train_metric = IoU(2)
    val_metric = IoU(2)
    best_acc = 0.0
    
    if start_from_current_accuracy:
        val_loss, val_acc = eval_step(model, eval_loader, criterions, device, val_metric)
        best_acc = val_acc

    best_model_wts = model.state_dict()
    val_acc_history = []
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_step(model, opt, train_loader, criterions, device, train_metric)      
        val_loss, val_acc = eval_step(model, eval_loader, criterions, device, val_metric)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'Best acc now: [{best_acc}]')        
        val_acc_history.append(val_acc)
        
        model.load_state_dict(best_model_wts)        
        
        print(f'Epoch [{epoch+1}/{n_epochs}] | Loss (train/val): {train_loss:.4f}/{val_loss:.4f} '+\
               f' | Acc (train/val): {train_acc:.4f}/{val_acc:.4f}')

    print(f'Final best accuracy [{best_acc}]')
    return model, val_acc_history


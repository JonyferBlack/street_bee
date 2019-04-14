import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.rcParams['figure.figsize'] = [30, 20]
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[preds[j]]}')

                ax.imshow(inputs.data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    fig.show()


def imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.rcParams['figure.figsize'] = [20, 20]
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def full_norm(input_):
    min_ = input_.min() * -1 
    input_ += min_
    max_ = input_.max()
    input_ *= 255/max_
    return input_


def draw_mask(input_, mask, threshold = 30):
    mask = full_norm(mask)
    ret, thresholded = cv2.threshold(mask.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    thresholded = np.expand_dims(thresholded, axis = 2)
    thresholded = thresholded.astype(np.uint8)
    thresholded = np.concatenate((thresholded, thresholded, thresholded), axis = 2)
    color_mask = thresholded.copy()
    shape = thresholded.shape
    color_mask[:,:,2] = np.zeros((shape[0],shape[1]), np.int8)
    norm_input = full_norm(input_).astype(np.uint8)
    output = cv2.addWeighted(norm_input, 1, color_mask, 0.3, 0)
    return output
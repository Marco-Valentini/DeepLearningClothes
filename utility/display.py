import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os


def display_items(df_reduced, number_of_items=10):
    """
    Given the reduced version of the catalogue, display the first 10 items
    :param df_reduced:
    :return:
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    indeces = list(df_reduced['ID'])[0:number_of_items]
    labels = list(df_reduced['Semantic_category'])[0:number_of_items]
    fig = plt.figure(figsize=(25,20))
    images=[]
    for idx in range(number_of_items):
        image = cv2.imread('../dataset/images/' + indeces[idx] + '.jpg')
        rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(rgbImg)

    for idx in range(number_of_items):
        ax = fig.add_subplot(2,int(number_of_items/2),idx+1,xticks=[],yticks=[])
        ax.set_title(labels[idx])
        plt.imshow(images[idx])
    plt.show()



def display_outfits(outfits: pd.DataFrame, number_of_outfits=5):
    """
    Given the dataframe containing the outfits, display the first number_of_outfits outfits
    :param outfits: the dataframe containing the outfits
    :param number_of_outfits: the number of outfits to display
    :return: None
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for i in range(number_of_outfits):
        images = []
        clothes_list = outfits.loc[i].values
        img1 = cv2.imread('../dataset/images/' + str(clothes_list[0]) + '.jpg')
        rgbImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        images.append(rgbImg1)
        img2 = cv2.imread('../dataset/images/' + str(clothes_list[1]) + '.jpg')
        rgbImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        images.append(rgbImg2)
        img3 = cv2.imread('../dataset/images/' + str(clothes_list[2]) + '.jpg')
        rgbImg3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        images.append(rgbImg3)
        img4 = cv2.imread('../dataset/images/' + str(clothes_list[3]) + '.jpg')
        rgbImg4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        images.append(rgbImg4)
        fig = plt.figure(figsize=(25, 20))
        for idx in range(4):
            ax = fig.add_subplot(1, int(4), idx+1, xticks=[], yticks=[])
            ax.set_title(clothes_list[idx])
            plt.imshow(images[idx])
        plt.show()

def display_single_outfit(ID_list: torch.Tensor):
    """
    Given the list of IDs of the items in the outfit, display the outfit
    :param ID_list: the list of IDs of the items in the outfit
    :return: None
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # passa 3 o 4 immagini a seconda che stiamo mostrando le 3 della query o le 4 predette
    images = []
    img1 = cv2.imread('../dataset/polyvore_outfits/images/' + str(ID_list[0]) + '.jpg')
    rgbImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    images.append(rgbImg1)
    img2 = cv2.imread('../dataset/polyvore_outfits/images/' + str(ID_list[1]) + '.jpg')
    rgbImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    images.append(rgbImg2)
    img3 = cv2.imread('../dataset/polyvore_outfits/images/' + str(ID_list[2]) + '.jpg')
    rgbImg3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    images.append(rgbImg3)
    if ID_list[3]:
        img4 = cv2.imread('../dataset/polyvore_outfits/images/' + str(ID_list[3]) + '.jpg')
        rgbImg4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        images.append(rgbImg4)
        fig = plt.figure(figsize=(25, 20))
        for idx in range(4):
            ax = fig.add_subplot(1, int(4), idx + 1, xticks=[], yticks=[])
            ax.set_title(ID_list[idx])
            plt.imshow(images[idx])
        plt.show()
    else:
        fig = plt.figure(figsize=(25, 20))
        for idx in range(3):
            ax = fig.add_subplot(1, int(4), idx + 1, xticks=[], yticks=[])
            ax.set_title(ID_list[idx])
            plt.imshow(images[idx])
        plt.show()

def display_predictions(predictions: torch.Tensor, catalogue: pd.DataFrame):
    """
    Given the predictions of the model, display the first 5 items
    :param predictions: the predictions of the model (a tensor containing the indexes of the items in the catalogue)
    :param num_items: the number of items to display
    :catalogue: the catalogue of items (a dataframe)
    :return: None
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    idxs = list(predictions.cpu().numpy())
    indices = catalogue['ID'].loc[idxs].values
    for i in range(len(idxs)):
        img = cv2.imread('../dataset/images/' + str(indices[i]) + '.jpg')
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgbImg)
        plt.show()



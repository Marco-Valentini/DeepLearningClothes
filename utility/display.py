import cv2
import matplotlib.pyplot as plt

def display_items(df_reduced):
    """
    Given the reduced version of the catalogue, display the first 10 items
    :param df_reduced:
    :return:
    """
    indeces = list(df_reduced['ID'])[0:10]
    labels = list(df_reduced['Semantic_category'])[0:10]
    fig = plt.figure(figsize=(25,20))
    images=[]
    for idx in range(10):
        image = cv2.imread('../dataset/images/' + indeces[idx] + '.jpg')
        rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(rgbImg)

    for idx in range(10):
        ax = fig.add_subplot(2,int(10/2),idx+1,xticks=[],yticks=[])
        ax.set_title(labels[idx])
        plt.imshow(images[idx])
    plt.show()



def display_outfits(outfits):
    for i in range(5):
        images = []
        clothes_list = outfits[i]['item_id']
        img1 = cv2.imread('../dataset/images/' + clothes_list[0] + '.jpg')
        rgbImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        images.append(rgbImg1)
        img2 = cv2.imread('../dataset/images/' + clothes_list[1] + '.jpg')
        rgbImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        images.append(rgbImg2)
        img3 = cv2.imread('../dataset/images/' + clothes_list[2] + '.jpg')
        rgbImg3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        images.append(rgbImg3)
        img4 = cv2.imread('../dataset/images/' + clothes_list[3] + '.jpg')
        rgbImg4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        images.append(rgbImg4)
        fig = plt.figure(figsize=(25, 20))
        for idx in range(4):
            ax = fig.add_subplot(1, int(4), idx+1, xticks=[], yticks=[])
            ax.set_title(clothes_list[idx])
            plt.imshow(images[idx])
        plt.show()

def display_single_outfit(ID_list):
    # passa 3 o 4 immagini a seconda che stiamo mostrando le 3 della query o le 4 predette
    images = []
    img1 = cv2.imread('../dataset/polyvore_outfits/images/' + ID_list[0] + '.jpg')
    rgbImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    images.append(rgbImg1)
    img2 = cv2.imread('../dataset/polyvore_outfits/images/' + ID_list[1] + '.jpg')
    rgbImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    images.append(rgbImg2)
    img3 = cv2.imread('../dataset/polyvore_outfits/images/' + ID_list[2] + '.jpg')
    rgbImg3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    images.append(rgbImg3)
    if ID_list[3]:
        img4 = cv2.imread('../dataset/polyvore_outfits/images/' + ID_list[3] + '.jpg')
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



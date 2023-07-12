import pandas as pd


def get_category_labels(dataframe: pd.DataFrame, catalogues: dict):
    """
    this function returns the labels for each category of each outfit in the training set
    :param dataframe:
    :param catalogues:
    :return:
    """
    category_labels = {
        'shoes': [],
        'tops': [],
        'accessories': [],
        'bottoms': []
    }

    for outfit in dataframe.values:  # for each outfit in the training set
        for i in range(len(outfit)):  # for each item in the outfit
            if i == 0:
                catalogue = catalogues['shoes']
                ID = outfit[i]
                # find the ID in the catalogue and get its index
                index = catalogue[catalogue['ID'] == ID].index[0]
                # append the index to the list of labels
                category_labels['shoes'].append(index)
            elif i == 1:
                catalogue = catalogues['tops']
                ID = outfit[i]
                index = catalogue[catalogue['ID'] == ID].index[0]
                category_labels['tops'].append(index)
            elif i == 2:
                catalogue = catalogues['accessories']
                ID = outfit[i]
                index = catalogue[catalogue['ID'] == ID].index[0]
                category_labels['accessories'].append(index)
            else:
                catalogue = catalogues['bottoms']
                ID = outfit[i]
                index = catalogue[catalogue['ID'] == ID].index[0]
                category_labels['bottoms'].append(index)
    return category_labels
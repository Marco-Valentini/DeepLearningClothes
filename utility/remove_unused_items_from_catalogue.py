def remove_unused_items_from_catalogue(catalogue,outfit_column):
    """
    Given a catalogue and an outfit dataframe, removes from the catalogue the items that are not used in the outfit dataframe.
    :param catalogue: catalogue to be analyzed.
    :param outfit_dataframe: outfit dataframe to be analyzed.
    :return: compacted catalogue.
    """
    drop_indeces = []
    # for each item in the catalogue, check if it is used in the outfit dataframe
    for i,item in enumerate(catalogue['ID'].values):
        if item not in outfit_column:
            # if it is not used, remove it from the catalogue
            drop_indeces.append(i)
    reduced_catalogue = catalogue.drop(drop_indeces,axis=0)
    return reduced_catalogue
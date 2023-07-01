def filter_outfit(outfit, conditions):
    """
    This function is used to filter only the outfits containing 1 top, 1 bottom, 1 pair of shoes and 1 accessory
    :param outfit: is a dictionary
    :param conditions: is a dataframe containing allowed items
    :return: filtered outfit or False (outfit doesn't contain required items or contains less than 4 elements)

    example of input outfit
    outfit = {'items': [{'item_id': '162715806', 'index': 1}, {'item_id': '171888747', 'index': 2}, {'item_id': '173096665', 'index': 3},
    {'item_id': '170904692', 'index': 4}, {'item_id': '172482221', 'index': 5}], 'set_id': '200742384'}
    example of conditions : list of acceptable IDs and semantic categories
    df_reduced
    """
    conditions_ID = list(conditions['ID'])  # explicit cast from Series to list
    conditions_cat = list(conditions['Semantic_category'])
    if len(outfit['items']) < 4:
        return False  # not enough items !
    new_outfit = []
    categories = []
    for item_dict in outfit['items']:
        item = item_dict['item_id']
        # condition to check if is possible to add the item to the outfit
        if item in conditions_ID:
            category = conditions_cat[conditions_ID.index(item)]
            if category not in categories:
                if len(new_outfit) < 4:
                    new_outfit.append(item)
                    categories.append(category)
                else:
                    return {'item_id': new_outfit, 'set_id': outfit['set_id']}
    if len(new_outfit) == 4:
        return {'item_id': new_outfit, 'set_id': outfit['set_id']}
    else:
        return False
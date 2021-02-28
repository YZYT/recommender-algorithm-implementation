# -*- coding:utf8-*-
"""
author: Dylon
"""
import os


def get_user_click(rating_file):
    """
    get user click list
    :param rating_file: input file
    :return: dict, key: userid, value: [itemid1, itemid2]
    """
    if not os.path.exists(rating_file):
        return {}
    fp = open(rating_file)
    print("GG")
    user_click = {}
    for line in fp:
        item = line.strip().split('::')
        if len(item) < 4:
            continue
        [userid, itemid, rating, timestamp] = item
        if float(rating) < 3.0:
            continue
        if userid not in user_click:
            user_click[userid] = []
        user_click[userid].append(itemid)

    fp.close()
    return user_click


def get_item_info(item_file):
    """
    get item info[title, genres]
    :param item_file: iteminfo file
    :return: a dict, key: itemid, value: [title, genre]
    """
    if not os.path.exists(item_file):
        return {}
    fp = open(item_file, 'r',encoding='utf-8')
    item_info = {}
    for line in fp:
        item = line.strip().split('::')
        if len(item) < 3:
            continue
        [itemid, title, genre] = item
        if itemid not in item_info:
            item_info[itemid] = [title, genre]
    fp.close()
    return item_info

# if __name__ == '__main__':
    # user_click = get_user_click('../data/ratings.dat')
    # print(len(user_click))
    # print(user_click["1"])

    # item_info = get_item_info('../data/movies.dat')
    # print(len(item_info))
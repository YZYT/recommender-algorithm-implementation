# -*- coding:utf8-*-
"""
item cf main algo
author: Dylon
"""
import sys
import util.reader as reader
sys.path.append('../util')


def item_sim(user_click):
    """

    :param user_click: dict, key: userid, value: [itemid1, itemid2]
    :return: dict key: itemid i, value: dict, value_key: itemid j, value_value: simscore
    """
    pass

    for user, itemlist in user_click():
        pass

def main_flow():
    """
    main flow of itemcf
    :return:
    """
    user_click = reader.get_user_click('../data/ratings.dat')
    sim_info = item_sim(user_click)
    recom_result = recom_result(sim_info, user_click)
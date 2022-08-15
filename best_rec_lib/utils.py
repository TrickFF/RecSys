import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / popularity['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
 
    # Уберем самые популярные товары (их и так купят)
    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    items_last_12 = data[data['week_no'] >= 48].item_id.tolist()
    items_older_12 = data[data['week_no'] < 48].item_id.tolist()
    items_not_sale_12 = list(set(items_older_12) - set(items_last_12))
    data = data[~data['item_id'].isin(items_not_sale_12)]

#     # Уберем не интересные для рекоммендаций категории (department)
#     if item_features is not None:
#         department_size = item_features.groupby(['department'])['item_id'].nunique().reset_index()
#         department_size = department_size.sort_values('item_id', ascending=False)
#         department_size.rename(columns={'item_id': 'items_num'}, inplace=True)
#         small_departments = department_size[department_size['items_num'] < 100].department.tolist()
#         items_in_small_departments = item_features[item_features['department'].isin(small_departments)].item_id.unique().tolist()
#         data = data[~data['item_id'].isin(items_in_small_departments)]    

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # Также уберем слишком дорогие товары
    data = data[((data['sales_value'] / data['quantity']) > 2) & ((data['sales_value'] / data['quantity']) < 200)]
    
    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
     
    # Заведем фиктивный item_id (если юзер не покупал товары из топ, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def postfilter_items(user_id, recommednations):
    pass
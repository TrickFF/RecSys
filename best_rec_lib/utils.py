import pandas as pd
import numpy as np
import gc
import math


def prefilter_items(data, take_n_popular=5000, item_features=None):
    
    '''
    Учитывая, что мы будем в пределах разумного максимизировать MAP@k, то популярные и непопулярные товары не фильтруем
    '''
#     week = data['week_no'].max()
#     popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
#     popularity['user_id'] = popularity['user_id'] / week
#     popularity.rename(columns={'user_id': 'avg_user_purchases_week'}, inplace=True)

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
#     top_notpopular = popularity.loc[popularity['avg_user_purchases_week'] < 0.02].item_id.tolist()
#     data = data[~data['item_id'].isin(top_notpopular)]

#     # Уберем самые популярные товары (их и так купят)
#     top_popular = popularity.loc[popularity['avg_user_purchases_week'] > 10].item_id.tolist()
#     data = data[~data['item_id'].isin(top_popular)]    
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    week = data['week_no'].max()
    items = data.item_id.tolist()
    items_last_12 = data[data['week_no'] > week - 48].item_id.tolist()
    items_not_sale_12 = list(set(items) - set(items_last_12))
    data = data[~data['item_id'].isin(items_not_sale_12)]

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = item_features.groupby(['department'])['item_id'].nunique().reset_index()
        department_size = department_size.sort_values('item_id', ascending=False)
        department_size.rename(columns={'item_id': 'items_num'}, inplace=True)
        small_departments = department_size[department_size['items_num'] < 30].department.tolist()
        items_in_small_departments = item_features[item_features['department'].isin(small_departments)].item_id.unique().tolist()
        data = data[~data['item_id'].isin(items_in_small_departments)]    

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

def lvl2Features(data_train, user_features, item_features):
    user_feats = user_features[['user_id', 'age_desc', 'income_desc']]
    item_feats = item_features[['item_id', 'department']]

    spam_features = data_train.merge(user_feats, on='user_id', how='left')
    spam_features = spam_features.merge(item_feats, on='item_id', how='left')

    del user_feats
    del item_feats

    gc.collect()
    
    ##### Фичи для пары user-товар #####
    # Среднее количество покупок пользователями по категориям
    weeks = spam_features.week_no.max()
    cat_num_week_buy = spam_features.groupby(['department'])[['sales_value']].count()
    cat_num_week_buy = cat_num_week_buy.rename(columns= {'sales_value': 'cat_num_week_buy'})
    cat_num_week_buy['cat_num_week_buy'] = cat_num_week_buy['cat_num_week_buy'] / weeks
    
    # средний чек всех пользователей по категории товара
    avg_check_cat_all_users = spam_features.groupby(['department'])[['sales_value']].mean()
    avg_check_cat_all_users = avg_check_cat_all_users.rename(columns= {'sales_value': 'avg_check_cat_all_users'})
    
    
    ##### Фичи для user #####
    # средний и максимальный чек пользователя
    check = spam_features.groupby(['user_id', 'basket_id'])[['sales_value']].sum()
    avg_check = check.groupby(['user_id'])[['sales_value']].mean()
    avg_check = avg_check.rename(columns= {'sales_value': 'avg_check'})
    max_check = check.groupby(['user_id'])[['sales_value']].max()
    max_check = max_check.rename(columns= {'sales_value': 'max_check'})

    del check
    gc.collect()
    
    # средний и максимальный чек пользователя по категориям
    check_cat = spam_features.groupby(['user_id', 'basket_id', 'department'])[['sales_value']].sum()
    avg_check_cat = check_cat.groupby(['user_id', 'department'])[['sales_value']].mean()
    avg_check_cat = avg_check_cat.rename(columns= {'sales_value': 'avg_check_cat'})
    max_check_cat = check_cat.groupby(['user_id', 'department'])[['sales_value']].max()
    max_check_cat = max_check_cat.rename(columns= {'sales_value': 'max_check_cat'})

    del check_cat
    gc.collect()
    
    ##### Фичи для товаров #####
    # мода возрастной категории, приобретающих товар
    age_mode = pd.DataFrame(spam_features.groupby('item_id')[['age_desc']].value_counts())
    age_mode = age_mode.rename(columns= {0: 'age_mode'})
    spam = list(age_mode.index)
    age_mode = {'item_id': [], 'age_mode': []}
    for i, el in enumerate(spam):
        if spam[i][0] != spam[i-1][0]:
            age_mode['item_id'].append(el[0])
            age_mode['age_mode'].append(el[1])

    age_mode = pd.DataFrame(age_mode, columns =['item_id', 'age_mode'])
    
    # мода категории дохода, приобретающих товар
    inc_mode = pd.DataFrame(spam_features.groupby('item_id')[['income_desc']].value_counts())
    inc_mode = inc_mode.rename(columns= {0: 'inc_mode'})
    spam = list(inc_mode.index)
    inc_mode = {'item_id': [], 'inc_mode': []}
    for i, el in enumerate(spam):
        if spam[i][0] != spam[i-1][0]:
            inc_mode['item_id'].append(el[0])
            inc_mode['inc_mode'].append(el[1])

    inc_mode = pd.DataFrame(inc_mode, columns =['item_id', 'inc_mode'])
    
    # Сумма покупок пользователем товара
    item_buy_sum = spam_features.groupby(['user_id', 'item_id'])[['sales_value']].sum()
    item_buy_sum = item_buy_sum.rename(columns= {'sales_value': 'item_buy_sum'})
    
    # Количество раз покупоки пользователем товара
    item_buy_count = spam_features.groupby(['user_id', 'item_id'])[['sales_value']].count()
    item_buy_count = item_buy_count.rename(columns= {'sales_value': 'item_buy_count'})
    
    # Общее количество единиц товара купленного пользователем
    item_buy_num_sum = spam_features.groupby(['user_id', 'item_id'])[['quantity']].sum()
    item_buy_num_sum = item_buy_num_sum.rename(columns= {'quantity': 'item_buy_num_sum'})
    
    # Доля заказов в которых есть айтем
    num_orders = spam_features.basket_id.nunique()
    orders_with_item = spam_features.groupby(['item_id'])[['basket_id']].count()
    orders_with_item = orders_with_item.rename(columns= {'basket_id': 'orders_with_item'})
    orders_with_item['orders_with_item'] = orders_with_item['orders_with_item'] / num_orders

    # Количество дней с момента последней покупки пользователем товара
    # Если не покупал, то 1000
    max_day = spam_features.day.max()
    user_last_buy_item = spam_features.groupby(['user_id', 'item_id'])[['day']].max()
    user_last_buy_item = user_last_buy_item.rename(columns= {'day': 'user_last_buy_item'})
    user_last_buy_item['user_last_buy_item'] = max_day - user_last_buy_item['user_last_buy_item']
    
    
    return age_mode, inc_mode, cat_num_week_buy, avg_check_cat_all_users, avg_check, max_check, avg_check_cat, max_check_cat, item_buy_sum, item_buy_count, item_buy_num_sum, orders_with_item, user_last_buy_item

def trainTestDf(data_train, targets_lvl_2, item_features, user_features):
    targets_lvl_2 = targets_lvl_2.merge(item_features, on='item_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(user_features, on='user_id', how='left')

    
    age_mode, inc_mode, cat_num_week_buy, avg_check_cat_all_users, avg_check, max_check, avg_check_cat, max_check_cat, item_buy_sum, item_buy_count, item_buy_num_sum, orders_with_item, user_last_buy_item = lvl2Features(data_train, user_features, item_features)
    
    targets_lvl_2 = targets_lvl_2.merge(age_mode, on='item_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(inc_mode, on='item_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(cat_num_week_buy, on='department', how='left')
    targets_lvl_2 = targets_lvl_2.merge(avg_check_cat_all_users, on='department', how='left')
    targets_lvl_2 = targets_lvl_2.merge(avg_check, on='user_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(max_check, on='user_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(avg_check_cat, on=['user_id', 'department'], how='left')
    targets_lvl_2 = targets_lvl_2.merge(max_check_cat, on=['user_id', 'department'], how='left')
    targets_lvl_2 = targets_lvl_2.merge(item_buy_sum, on=['user_id', 'item_id'], how='left')
    targets_lvl_2 = targets_lvl_2.merge(item_buy_count, on=['user_id', 'item_id'], how='left')
    targets_lvl_2 = targets_lvl_2.merge(item_buy_num_sum, on=['user_id', 'item_id'], how='left')
    targets_lvl_2 = targets_lvl_2.merge(orders_with_item, on='item_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(user_last_buy_item, on=['user_id', 'item_id'], how='left')
    
    ##### Делим данные на обучающую выборку и таргет #####
    # Заполняем пропуски
    targets_lvl_2 = targets_lvl_2.fillna('None/Unknown')
    targets_lvl_2.loc[targets_lvl_2['cat_num_week_buy'] == 'None/Unknown', 'cat_num_week_buy'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['avg_check_cat_all_users'] == 'None/Unknown', 'avg_check_cat_all_users'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['avg_check'] == 'None/Unknown', 'avg_check'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['max_check'] == 'None/Unknown', 'max_check'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['avg_check_cat'] == 'None/Unknown', 'avg_check_cat'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['max_check_cat'] == 'None/Unknown', 'max_check_cat'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['item_buy_sum'] == 'None/Unknown', 'item_buy_sum'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['item_buy_count'] == 'None/Unknown', 'item_buy_count'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['item_buy_num_sum'] == 'None/Unknown', 'item_buy_num_sum'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['orders_with_item'] == 'None/Unknown', 'orders_with_item'] = 0.0
    targets_lvl_2.loc[targets_lvl_2['user_last_buy_item'] == 'None/Unknown', 'user_last_buy_item'] = 1000.0
    
    targets_lvl_2["avg_check_cat"] = targets_lvl_2.avg_check_cat.astype(np.float64)
    targets_lvl_2["max_check_cat"] = targets_lvl_2.max_check_cat.astype(np.float64)
    targets_lvl_2["item_buy_sum"] = targets_lvl_2.item_buy_sum.astype(np.float64)
    targets_lvl_2["item_buy_count"] = targets_lvl_2.item_buy_count.astype(np.float64)
    targets_lvl_2["item_buy_num_sum"] = targets_lvl_2.item_buy_num_sum.astype(np.float64)
    targets_lvl_2["orders_with_item"] = targets_lvl_2.orders_with_item.astype(np.float64)
    targets_lvl_2["user_last_buy_item"] = targets_lvl_2.user_last_buy_item.astype(np.float64)
    
    X_train = targets_lvl_2.drop('target', axis=1)
    y_train = targets_lvl_2[['target']]
    
    cat_feats = X_train.columns[2:17].tolist()
    X_train[cat_feats] = X_train[cat_feats].astype('category')
    
    return X_train, y_train, cat_feats


def trainValLvl1Split(data, test_user_ids):
    
    def test_baskets(x):
        start = math.ceil(x['basket_id_len']*0.1)
        return x['basket_id'][len(x['basket_id']) - start:]
    
    data = data.loc[data['user_id'].isin(test_user_ids)]
    data_spam = data.groupby('user_id')['basket_id'].unique().reset_index()
    data_spam['basket_id_len'] = data_spam['basket_id'].apply(lambda x: len(x))
    data_spam['test_basket_id'] = data_spam.apply(lambda x: test_baskets(x), axis=1)
    
    data_val_lvl_1_list = []
    for i, row in data_spam.iterrows():
        data_val_lvl_1_list.extend(row['test_basket_id'])
    
    data_val_lvl_1 = data.loc[data['basket_id'].isin(data_val_lvl_1_list)]
    
    del data_spam
    gc.collect()
    
    data_train_lvl_1 = data.loc[~data['basket_id'].isin(data_val_lvl_1_list)]
    
    return data_train_lvl_1, data_val_lvl_1


def getRecommendationsLvl2(preds_lvl_2, x, recommender, N=5):
    spam = preds_lvl_2.loc[preds_lvl_2['user_id'] == x]
    
    if spam.shape[0] > 0:
        # если user_id есть в результатах классификатора - берем топ5 рекомендаций
        spam = spam.sort_values('pred_proba', ascending=False).head(N)
        spam = list(spam.item_id)
    else:
        # если user_id нет в результатах классификатора - берем топ5 популярных товаров
        spam = recommender._extend_with_top_popular([], N=5)
    
    return spam


def postfilter_items(user_id, recommednations):
    pass
import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Классификатор
from catboost import CatBoostClassifier

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# Для формирования датасета
from best_rec_lib.utils import trainTestDf

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    weighting: string
        Тип взвешивания, один из вариантов None, "bm25", "tfidf"
    fake_id: int
        идентификатор, которым заменялись редкие объекты, можно передать None?если такого объекта нет
    """

    def __init__(self, data, weighting="bm25", fake_id = 999999):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != fake_id]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != fake_id]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.fake_id = fake_id
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        
        self.sparse_user_items = csr_matrix(self.user_item_matrix).tocsr()
        
        if weighting == "bm25":
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T.tocsr()   
        elif weighting == "tfidf":
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T.tocsr()
        else:
            self.user_item_matrix = self.sparse_user_items

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробоват ддругие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=-1)
        own_recommender.fit(user_item_matrix)

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=350, regularization=0.01, iterations=1, num_threads=-1):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(user_item_matrix)

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=3)  # Товар похож на себя -> рекомендуем 2 товара
        
        recs = list(recs[0][1:])        
        if self.itemid_to_id[self.fake_id] in recs:
            recs.remove(self.itemid_to_id[self.fake_id])
        top_rec = recs[0]
        
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            top_popular = [rec for rec in self.overall_top_purchases[:N] if rec not in recommendations]
            recommendations.extend(top_popular)
            recommendations = recommendations[:N]

        return recommendations
    
    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = model.recommend(userid=self.userid_to_id[user],
            user_items=self.sparse_user_items[self.userid_to_id[user]],
            N=N,
            filter_already_liked_items=False,
            filter_items=[self.itemid_to_id[self.fake_id]],
            recalculate_user=True)
        
        mask = res[1].argsort()[::-1]
        
        res = [self.id_to_itemid[rec] for rec in res[0][mask]]
        res = self._extend_with_top_popular(res, N=N)        
        res = res[:N]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    
    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        res = []
        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N).item_id.tolist()
        
        for i, item in enumerate(top_users_purchases):
            spam = self._get_similar_item(item)
            res.append(spam)
        
        res = self._extend_with_top_popular(res, N=N)
 
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        res = []
        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = similar_users[0]
        similar_users = similar_users[1:]   # удалим юзера из запроса
        
        # Берем топ купленых товаров
        for user in similar_users:
            spam_id = self.id_to_userid[user]
            
            top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == spam_id].head(1)              
            res.extend(top_users_purchases.item_id)
            
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
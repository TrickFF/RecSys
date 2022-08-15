import numpy as np
import pandas as pd
def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    precision = flags.sum() / len(recommended_list)
    
    
    return precision


def money_precision_at_k(recommended_list, bought_list, prices, k=5):
        
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    precision = (flags*prices_recommended).sum() / prices_recommended.sum()
     
    return precision

def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def money_recall_at_k(recommended_list, bought_list, prices, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
        
    # Формируем список prices_recommended из массива prices
    spam = []
    for el in recommended_list:
        if el in prices:
            spam.append(prices[el])
            
    prices_recommended = np.array(spam)
    
    # Формируем список prices_bought из массива prices
    spam = []
    for el in bought_list:
        if el in prices:
            spam.append(prices[el])
            
    prices_bought = np.array(spam)

    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
       
    flags = np.isin(recommended_list, bought_list)
        
    recall = (flags * prices_recommended).sum() / prices_bought.sum()
    
    return recall

def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(k):
        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
            sum_ += p_k
            
    result = sum_ / k
    
    return result
import os
import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''

    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()
        self.matching_table = Olist().get_matching_table()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        orders = self.data['orders'].copy()
        orders_d = orders[orders['order_status'] == 'delivered'].copy()
        # difference between order purchase timestamp and order delivered time
        orders_d['order_purchase_timestamp'] = pd.to_datetime(orders_d['order_purchase_timestamp'])
        orders_d['order_delivered_customer_date'] = pd.to_datetime(orders_d['order_delivered_customer_date'])
        orders_d['order_estimated_delivery_date'] = pd.to_datetime(orders_d['order_estimated_delivery_date'])
        orders_d['wait_time'] = (orders_d['order_delivered_customer_date'] - orders_d['order_purchase_timestamp']).dt.days
        orders_d['expected_wait_time'] = (orders_d['order_estimated_delivery_date'] - orders_d['order_purchase_timestamp']).dt.days
        orders_d['delay_vs_expected'] = (orders_d['wait_time'] - orders_d['expected_wait_time']).apply(lambda x: x if x and x > 0 else 0)


        return orders_d[['order_id', 'wait_time', 'expected_wait_time',
                        'delay_vs_expected', 'order_status']]


    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()
        reviews['dim_is_five_star'] = reviews['review_score'].apply(lambda x:
            1 if x and x == 5 else 0)
        reviews['dim_is_one_star'] = reviews['review_score'].apply(
            lambda x: 1 if x and x == 1 else 0)

        return reviews[[
            'order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score'
        ]]


    def get_number_products(self):
        """
        Returns a DataFrame with:
        order_id, number_of_products
        """
        order_items = self.data['order_items'].copy()
        occurences = order_items.groupby('order_id').count()[['product_id']]

        occurences.columns = ['number_of_products']

        return occurences


    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        order_items = self.data['order_items'].copy()
        occurences = order_items.groupby('order_id').nunique()[['seller_id']]
        occurences.columns = ['number_of_sellers']

        return occurences

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        order_items = self.data['order_items'].copy()
        occurences = order_items.groupby('order_id').sum()

        return occurences[['price', 'freight_value']]

    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with order_id
        and distance_seller_customer
        """
        location = self.data['geolocation'].copy().groupby('geolocation_zip_code_prefix').first()
        sellers = self.data['sellers'].copy()
        customers = self.data['customers'].copy()


        ev = self.matching_table.merge(customers, on = 'customer_id').merge(sellers, on = 'seller_id').merge(location, left_on = 'seller_zip_code_prefix', right_on = 'geolocation_zip_code_prefix').rename(columns = {'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}) \
        .merge(location, left_on = 'customer_zip_code_prefix', right_on = 'geolocation_zip_code_prefix').rename(columns = {'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'})

        ev['distance_seller_customer'] = ev.apply(
            lambda x: haversine_distance(x['seller_lng'], x['seller_lat'], x[
                'customer_lng'], x['customer_lat']),
            axis=1)
        # if there are multiple sellers, take the mean
        return ev[['order_id','distance_seller_customer']].groupby('order_id').mean()

    def get_training_data(self, is_delivered=True, with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_products', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        wait_time = self.get_wait_time()
        review_score = self.get_review_score()
        number_products = self.get_number_products()
        number_sellers = self.get_number_sellers()
        price_and_freight = self.get_price_and_freight()

        m1 = pd.merge(wait_time, review_score, on = 'order_id')
        m2 = pd.merge(m1, number_products, on = 'order_id')
        m3 = pd.merge(m2, number_sellers, on = 'order_id')
        m4 = pd.merge(m3, price_and_freight, on = 'order_id')
        m4.dropna(inplace = True)

        if with_distance_seller_customer:
            distance_seller_customer = self.get_distance_seller_customer()
            return pd.merge(m4, distance_seller_customer, on = 'order_id')

        return m4

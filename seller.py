import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy()

        return sellers[['seller_id', 'seller_city', 'seller_state']]

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """

        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].copy()
        orders_delievered = orders[orders['order_status'] == 'delivered'].copy()

        order_sellers = order_items.merge(orders_delievered, on = 'order_id')
        order_sellers['order_delivered_carrier_date'] = pd.to_datetime(
            order_sellers['order_delivered_carrier_date'])
        order_sellers['order_delivered_customer_date'] = pd.to_datetime(
            order_sellers['order_delivered_customer_date'])
        order_sellers['shipping_limit_date'] = pd.to_datetime(order_sellers['shipping_limit_date'])
        order_sellers['order_purchase_timestamp'] = pd.to_datetime(order_sellers['order_purchase_timestamp'])

        order_sellers['delay_to_carrier'] = list(
            map(
                lambda x: 0
                if x < 0 else x,
                (order_sellers['order_delivered_carrier_date'] -
                order_sellers['shipping_limit_date']) / np.timedelta64(24, 'h')))

        order_sellers['wait_time'] = (
            order_sellers['order_delivered_customer_date'] -
            order_sellers['order_purchase_timestamp']) / np.timedelta64(
                24, 'h')
        order_sellers = order_sellers[['seller_id', 'delay_to_carrier', 'wait_time']].groupby('seller_id').mean()

        return order_sellers

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale'
        """
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].copy()
        orders_delievered = orders[orders['order_status'] == 'delivered'].copy()

        order_sellers = order_items.merge(orders_delievered, on = 'order_id')
        first_date = order_sellers[['seller_id', 'order_purchase_timestamp']].groupby('seller_id').min().rename(columns = {'order_purchase_timestamp': 'date_first_sale'}).copy()
        last_date = order_sellers[['seller_id', 'order_purchase_timestamp']].groupby('seller_id').max().rename(columns = {'order_purchase_timestamp': 'date_last_sale'}).copy()

        return first_date.merge(last_date, on = 'seller_id')

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score'
        """
        def review_score_cost(score):
            cost_dir = {1: 100, 2: 50, 3:40, 4:0, 5: 0}
            return cost_dir[score]

        orders = self.order.get_training_data()
        order_items = self.data['order_items'].copy()[['order_id', 'seller_id']].drop_duplicates()
        order_reviews = order_items.merge(orders, on = 'order_id')
        order_reviews['review_cost'] = order_reviews['review_score'].apply(
            review_score_cost)

        five_star = order_reviews[['seller_id', 'dim_is_five_star']].copy().groupby('seller_id').mean().rename(columns = {'dim_is_five_star': 'share_of_five_stars'})
        one_star = order_reviews[['seller_id', 'dim_is_one_star']].copy().groupby('seller_id').mean().rename(columns = {'dim_is_one_star': 'share_of_one_stars'})

        order_reviews = order_reviews.groupby('seller_id',
                              as_index=False).agg({
                                  'dim_is_one_star' : 'mean',
                                  'dim_is_five_star': 'mean',
                                  'review_score': 'mean',
                                  'review_cost': 'sum'
                                  })
        order_reviews.columns = [
            'seller_id', 'share_of_one_stars', 'share_of_five_stars',
            'review_score', 'review_cost'
        ]

        return order_reviews


    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        # Hint: Here, you cannot start from the `matching_table`
        order_items = self.data['order_items'].copy()
        #number of unique orders
        norders = order_items[['seller_id', 'order_id']].groupby('seller_id').nunique().rename(columns = {'order_id': 'n_orders'}).copy()
        quantity = order_items[['seller_id', 'order_id']].groupby('seller_id').count().rename(columns = {'order_id': 'quantity'}).copy()
        quantity_df = norders.merge(quantity, on = 'seller_id')
        quantity_df['quantity_per_order'] = quantity_df['quantity'] / quantity_df['n_orders']

        return quantity_df

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        order_items = self.data['order_items'].copy()
        return order_items[['seller_id', 'price']].groupby('seller_id').sum().rename(columns = {'price': 'sales'})

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'share_of_one_stars',
        'share_of_five_stars', 'review_score', 'n_orders', 'quantity',
        'quantity_per_order', 'sales']
        """
        seller_features = self.get_seller_features()
        seller_delay_wait_time = self.get_seller_delay_wait_time()
        active_dates = self.get_active_dates()
        review_score = self.get_review_score()
        quantity = self.get_quantity()
        sales = self.get_sales()

        return seller_features.merge(seller_delay_wait_time, on = 'seller_id').merge(active_dates, on = 'seller_id') \
            .merge(review_score, on = 'seller_id').merge(quantity, on = 'seller_id').merge(sales, on = 'seller_id')

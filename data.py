import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """

        csv_path = os.path.join(os.path.dirname(__file__), "../data/csv/")
        file_names = [f for f in os.listdir(csv_path) if not f.startswith('.')]
        key_names = [
            f.strip(".csv").strip("olist").strip("dataset").strip("_")
            for f in file_names
        ]


        data = {}
        for i in range(len(key_names)):
            data[key_names[i]] = pd.read_csv(os.path.join(csv_path, file_names[i]))
        return data

    def get_matching_table(self):
        """
        This function returns a matching table between
        columns [ "order_id", "review_id", "customer_id", "product_id", "seller_id"]
        """
        data = self.get_data()
        orders = data['orders'][['order_id', 'customer_id']]
        reviews = data['order_reviews'][['review_id', 'order_id']]
        order_items = data['order_items'][[
            'product_id', 'order_id', 'seller_id'
        ]]
        matching_table = pd.merge(pd.merge(order_items,reviews,how="outer", on='order_id'), orders, how = 'outer', on= 'order_id')
        matching_table.drop_duplicates(keep='first', inplace=True)

        return matching_table

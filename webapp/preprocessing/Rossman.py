import os
import pickle
import pandas as pd
import math
import numpy as np
import datetime
import inflection

class RossmanPreprocessing( object ):
    def __init__(self):
        self.home_path =''
        self.competition_distance_scaler        = pickle.load(open( self.home_path + 'parameters/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler      = pickle.load(open(self.home_path + 'parameters/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler             = pickle.load(open(self.home_path + 'parameters/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                        = pickle.load(open(self.home_path + 'parameters/year_scaler.pkl', 'rb'))
        self.store_type_encoding                = pickle.load(open(self.home_path + 'parameters/store_type_encoding.pkl', 'rb'))

    def data_cleaning(self, df_raw_1):                                     # Removed ['sales', 'customers'], as they are not used for prediction
        old_cols = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                'Promo2SinceYear', 'PromoInterval']

        # Changing columns to snakecase
        snakecase = lambda x: inflection.underscore(x)
        new_cols = list(map(snakecase, old_cols))

        # Rename columns
        df_raw_1.columns = new_cols

        # change date to datetime64
        df_raw_1['date'] = pd.to_datetime(df_raw_1['date'])

        # 1.5. Fillout NA
        df_raw_1['competition_distance'].fillna(200000.0, inplace=True)

        df_raw_1['competition_open_since_month'] = (df_raw_1.apply(lambda x: x['date'].month 
                                                                if math.isnan(x['competition_open_since_month']) 
                                                                else x['competition_open_since_month'] , axis=1))

        df_raw_1['competition_open_since_year'] = (df_raw_1.apply(lambda x: x['date'].year 
                                                                if math.isnan(x['competition_open_since_year']) 
                                                                else x['competition_open_since_year'] , axis=1))

        df_raw_1['promo2_since_week'] = (df_raw_1.apply(lambda x: x['date'].week  
                                                        if math.isnan(x['promo2_since_week']) 
                                                        else x['promo2_since_week'] , axis=1))

        df_raw_1['promo2_since_year'] = (df_raw_1.apply(lambda x: x['date'].year  
                                                        if math.isnan(x['promo2_since_year']) 
                                                        else x['promo2_since_year'] , axis=1))

        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Set', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df_raw_1['promo_interval'].fillna(0, inplace=True)

        df_raw_1['month_map'] = df_raw_1['date'].dt.month.map(month_map)

        df_raw_1['is_promo'] = df_raw_1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        # 1.6 Revisiting Columns Types and Missing values again 
        df_raw_1['competition_open_since_month'] = df_raw_1['competition_open_since_month'].astype(int)
        df_raw_1['competition_open_since_year'] = df_raw_1['competition_open_since_year'].astype(int)

        df_raw_1['promo2_since_week'] = df_raw_1['promo2_since_week'].astype(int)
        df_raw_1['promo2_since_year'] = df_raw_1['promo2_since_year'].astype(int)

        return df_raw_1
    
    def feature_engineering(self, df2):
        #year
        df2['year'] = df2['date'].dt.year

        #month
        df2['month'] = df2['date'].dt.month

        #day
        df2['day'] = df2['date'].dt.day

        #week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week

        # year - week (mask)
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        #competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)

        # versão simplificada do script da aula
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).dt.days

        # converte ano-semana da promo em data da promo e calcula a diferença entre a data da promo e a data do registro de venda 
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).dt.days

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # 2.5. Row filtering 
        df2 = df2[df2['open'] != 0]

        # 2.6. Column selection
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)

        return df2
    
    def data_preparation(self, df5):
        # 5.2 Rescaling 
        # competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)

        # competition_time_month 
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values)

        # promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)

        # year
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        # 5.3.1 Encoding
        # state_holiday - One hot encoding
        df5 = pd.get_dummies(df5, prefix= ['state_holiday'], columns= ['state_holiday'])

        # store_type - Label Encoding
        df5['store_type'] = self.store_type_encoding.fit_transform(df5['store_type'])

        # assortment - Ordinal enconding (manual)
        assortment_dict = {'basic': 1,
                        'extra': 2,
                        'extended': 3}

        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # 5.3.3 Nature transformation
        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2*np.pi / 12) ))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2*np.pi / 12) ))

        # day 
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2*np.pi / 30) ))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2*np.pi / 30) ))

        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2*np.pi / 52) ))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2*np.pi / 52) ))

        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2*np.pi / 7) ))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2*np.pi / 7) ))

        cols_selected = ['store','promo','store_type','assortment','competition_distance','competition_open_since_month','competition_open_since_year','promo2',
                         'promo2_since_week','promo2_since_year','competition_time_month','promo_time_week','month_cos','month_sin','day_sin','day_cos',
                         'week_of_year_cos','week_of_year_sin','day_of_week_sin','day_of_week_cos']

        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # Prediction
        pred = model.predict(test_data)

        # Joining results
        original_data['predictions'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')

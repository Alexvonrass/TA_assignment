from feature_engineering import *

class ProcessingAndModel():
    
    def __init__(self, events_csv, geo_csv, lgb_params):
        self.eve = pd.read_csv(events_csv, parse_dates=['ts','date_from', 'date_to'], dtype={
        'event_type':'category', 'origin':'str', 'destination':'str', 'user_id':'category'
        , 'num_adults':np.int32, 'num_children':np.int32})
        self.geo = pd.read_csv(geo_csv).groupby('iata_code').agg({'lat':'mean', 'lon':'mean'}).reset_index()
        self.lgb_params = np.load(lgb_params, allow_pickle=True).item()
        
        # These are functions that I import and use to engineer features
        self.feat_eng_steps = [feat_eng_time_cat, feat_eng_time_delta
                               ,feat_eng_user_searches_tw, feat_eng_origin_searches_tw, feat_eng_total_searches_tw]
        
        # Time windows that i used to create new features
        self.time_windows = ['300s', '1h', '2h', '6h', '1d']

        # These are the features i ended up using in the final model
        self.features = ['num_adults', 'distance_km', 'trip_length', 'total_target_mean_last_300s_ratio',
                 'origin_int', 'dest_int', 'user_searches_last_1d', 'origin_target_mean_last_2h',
                 'total_searches_last_300s_ratio', 'total_target_mean_last_300s', 'total_searches_last_6h']

        # How many days we iteratively use for validation
        self.val_days = 3

        # Upsampling parameter
        self.upsampling = 5
        
        
    def feat_eng_build_model(self):
        # Create the target variable, equals to 1 if the event is 'book'
        self.eve['target']=np.int32(self.eve['event_type']=='book')
        self.eve.sort_values(by = 'ts', inplace=True)
        
        # Adding the distance between origin and destination as a feature
        self.eve = feat_eng_distance(self.eve, self.geo)
        
        # Feature engineering, some steps require a 'time_window' argument
        # , those have '_tw' in their name
        for step in self.feat_eng_steps:
            if '_tw' in step.__name__:
                self.eve = step(self.eve, self.time_windows)
            else:
                self.eve = step(self.eve)
                
        print("""Dataframe shape is {}, it has {} unique user_ids, and
         there are {} distinct days in the dataset""".format(
        self.eve.shape, self.eve['user_id'].nunique(), (self.eve['ts'].max() - self.eve['ts'].min()).days))
                
        # Drop the first day since time-window based features will be NA for these
        self.eve = self.eve[self.eve['ts'].dt.date > self.eve['ts'].min().date()+pd.Timedelta('1d')]
        
        # Use several last days for validation iteratively, ie use all data to predict the next day
        for val_day in self.eve['ts'].dt.date.unique()[-self.val_days:]:
            # Split the data into train and val, upsample positive examples in train but not val
            train = self.eve[(self.eve['ts'].dt.date < val_day)].copy()
            train =  pd.concat([train]+[train[train['target']==1].copy()]*self.upsampling, axis=0)
            val = self.eve[(self.eve['ts'].dt.date == val_day)].copy()
            
            # Encode origin and dest as categories separately
            train['origin'] = train['origin'].astype('category')
            train['origin_int'] = train['origin'].cat.codes
            val['origin'] = pd.Categorical(val['origin'], train['origin'].cat.categories)
            val['origin_int'] = val['origin'].cat.codes

            train['destination'] = train['destination'].astype('category')
            train['dest_int'] = train['destination'].cat.codes
            val['destination'] = pd.Categorical(val['destination'], train['destination'].cat.categories)
            val['dest_int'] = val['destination'].cat.codes
            
            # Pass names of categorical features to lgb
            cat = ['hour', 'day_of_week', 'origin_int', 'dest_int']
            cat = [c for c in cat if c in self.features]
            
            # Create datasets and train the model, validate and store the last model in the self.model attr
            lgbset_train = lgb.Dataset(train[self.features], train['target'], feature_name=self.features, categorical_feature=cat)
            lgbset_val = lgb.Dataset(val[self.features], val['target'], feature_name=self.features, categorical_feature=cat)
            bst = lgb.train(self.lgb_params, lgbset_train, valid_sets = [lgbset_train, lgbset_val]
            , num_boost_round = 50, verbose_eval=50)
            self.model = bst
        
           
    # This was not part of the assignment, but in production we would use 
    # the model and parameters stored in this object to score new data,
    # for example by saving/loading it as pickle
    def score(self, new_events_csv):
        pass
    
    
model = ProcessingAndModel(r'input/events.csv', r'input/iata.csv', 'lgb_params_v0.npy')
model.feat_eng_build_model()
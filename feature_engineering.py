from utils import *

@timeit
def feat_eng_distance(df, geo_df):
    """This function calculates a distance matrix of all possible permutations of iata_codes
    and adds it to the original dataframe as a separate column"""
    
    assert(df['origin'].isin(geo_df['iata_code']).all()&df['destination'].isin(
        geo_df['iata_code']).all())
    
    # This list comprehension creates all possible permutations of iata_code, lat and lon,
    # Calculates distances and stores them as a dataframe
    iata_pairs = pd.DataFrame([(x[0], y[0], geopy.distance.geodesic((x[1], x[2]),
     (y[1], y[2])).km) for (x, y) in permutations(zip(
         geo_df['iata_code'], geo_df['lat'], geo_df['lon']), 2)])
    iata_pairs.columns = ['origin', 'destination', 'distance_km']
    
    df = df.merge(iata_pairs, on = ['origin', 'destination'], how = 'left')
    return df
    

@timeit
def feat_eng_time_cat(df):
    """This function takes a dataframe and adds time-based columns like day of week"""
    #These are probably quite important
    df['day_of_week'] = df['ts'].dt.dayofweek  # Can't use this since the dataset is only 2 weeks and this overfits a lot
    df['hour'] = df['ts'].dt.hour
    
    #This might be important if people are more likely to book trips after getting their wages (for example at the end of the month)
    df['day'] = df['ts'].dt.day # Can't use this either, same reason
    
    #These are probably not needed, but doesn't hurt to check, maybe some types of fraud? 
    df['minute'] = df['ts'].dt.minute
    df['second'] = df['ts'].dt.second
    return df


@timeit
def feat_eng_time_delta(df):
    """This function takes a dataframe and adds length of stay and time until the trip"""
    df['time_to_trip'] = (df['date_from'] - df['ts']).dt.days # Bad feature for the same reason as day of week
    df['trip_length'] = (df['date_to'] - df['date_from']).dt.days
    return df


### Next three functions use pandas' groupby+rolling to extract information from different time windows
@timeit
def feat_eng_user_searches_tw(df, time_windows = ['1h', '1d']):
    """This function adds features based on each user's recent behavior"""
    for tw in time_windows:
        df['user_searches_last_'+tw] = df.set_index('ts').groupby('user_id').rolling(tw, closed='left')['target'].count().fillna(0).values
    return df


@timeit
def feat_eng_origin_searches_tw(df, time_windows = ['1h', '1d']):
    """This function calculates features for each origin 
    based on recent behavior"""
    for tw in time_windows:
        df['origin_searches_last_'+tw] = df.set_index('ts').groupby('origin').rolling(tw, closed='left')['target'].count().fillna(0).values
        df['origin_target_mean_last_'+tw] = df.set_index('ts').groupby('origin').rolling(tw, closed='left')['target'].mean().fillna(0).values
    return df


@timeit
def feat_eng_total_searches_tw(df, time_windows = ['300s', '1h', '2h', '6h', '1d']):
    """This function calculates features based on the entire dataset's recent behavior - 
    Amount of searches, how many recent searches ended in bookings and ratios compared to
    previous similar time window"""

    for tw in time_windows:
        df['total_searches_last_'+tw] = df.set_index('ts').rolling(tw, closed='left')['target'].count().fillna(0).values
        tmp = df.set_index('ts').rolling(str(int(tw[:-1])*2) + tw[-1], closed='left')['target'].count().fillna(0).values
        df['total_searches_last_'+tw+'_ratio'] = df['total_searches_last_'+tw]/(tmp)
        df['total_searches_last_'+tw+'_ratio'].fillna(0, inplace=True)
        df['total_target_mean_last_'+tw] = df.set_index('ts').rolling(tw, closed='left')['target'].sum().fillna(0).values
        tmp = df.set_index('ts').rolling(str(int(tw[:-1])*2) + tw[-1], closed='left')['target'].sum().fillna(0).values
        df['total_target_mean_last_'+tw+'_ratio'] = df['total_target_mean_last_'+tw]/(tmp)
        df['total_target_mean_last_'+tw+'_ratio'].fillna(0, inplace=True)
    return df
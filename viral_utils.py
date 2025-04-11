import pandas as pd
import numpy as np
from epiweeks import Week
import epiweeks as epi
import os


class ViralActivityUtils:
    map_level = {'minimal': 0, 'low': 1, 'moderate': 2, 'high': 3, 'very high': 4}

    @staticmethod
    def conv_epiweek(x):
        week = Week.fromdate(pd.to_datetime(x))
        return f"{week.year}EW{week.week:02}"

    @staticmethod
    def conv_epiweek_stdate(dt):
        return pd.to_datetime(epi.Week.fromdate(pd.to_datetime(dt)).startdate().strftime('%Y-%m-%d'))

    @staticmethod
    def get_levels(val, typ='cat'):
        if np.isnan(val):
            return np.nan
        if val < 1.5:
            cat, cat_val = 'minimal', 1
        elif val < 3:
            cat, cat_val = 'low', 2
        elif val < 4.5:
            cat, cat_val = 'moderate', 3
        elif val < 8:
            cat, cat_val = 'high', 4
        else:
            cat, cat_val = 'very high', 5
        return cat if typ == 'cat' else cat_val

    @staticmethod
    def get_baselines(basedf):
        b = basedf['log_viral_load'].quantile(q=0.1)
        sd = basedf['log_viral_load'].std()
        return b, sd

    @staticmethod
    def get_avl_dates(dates, date_range_min, date_range_max):
        avl_dates = [dt for dt in dates if date_range_min <= dt < date_range_max]
        return avl_dates, len(avl_dates)


class WastewaterProcessor:
    def __init__(self, df):
        self.df = df

    def check_missing_dates_freq(self):
        rmdf = pd.DataFrame()
        for s in self.df.geo_value.unique():
            temp = self.df[self.df.geo_value == s].copy()
            temp = temp.set_index('EW_stdate').asfreq('W-SUN').reset_index()
            temp['geo_value'] = temp['geo_value'].fillna(s)
            temp['region'] = temp['region'].fillna(temp.region.unique()[0])
            temp['geo_res'] = temp['geo_res'].ffill()
            rmdf = pd.concat([rmdf, temp])
        return rmdf

    def get_VAL(self, sdf):
        dates = pd.to_datetime(sdf.EW_stdate.unique())
        for dt in dates:
            if 7 <= dt.month <= 12:
                ref_min_date = pd.to_datetime(f'{dt.year}-01-01')
                ref_max_date = pd.to_datetime(f'{dt.year}-07-01')
            else:
                ref_min_date = pd.to_datetime(f'{dt.year - 1}-07-01')
                ref_max_date = pd.to_datetime(f'{dt.year - 1}-12-31')

            date_range_min = dates.min()
            date_range_max = dt

            if date_range_min <= ref_min_date and date_range_max > ref_max_date:
                date_range_min, date_range_max = ref_min_date, ref_max_date
            elif date_range_min >= ref_min_date and date_range_min + pd.Timedelta(weeks=26) < date_range_max:
                date_range_max = date_range_min + pd.Timedelta(weeks=26)
            elif date_range_min >= ref_min_date and date_range_min + pd.Timedelta(weeks=26) > date_range_max:
                if (date_range_max - date_range_min).days // 7 < 6:
                    continue

            avl_dates, avl_len = ViralActivityUtils.get_avl_dates(dates, date_range_min, date_range_max)
            basedf = sdf[sdf.EW_stdate.isin(avl_dates)]

            b, sd = ViralActivityUtils.get_baselines(basedf)
            
            mask = sdf.EW_stdate == dt

            sdf.loc[mask, 'b'] = b
            sdf.loc[mask, 'sd'] = sd
            sdf.loc[mask, 'eta'] = (sdf.loc[mask, 'log_viral_load'] - b) / sd
            sdf.loc[mask, 'viral_activity_level'] = np.exp(sdf.loc[mask, 'eta'])
            sdf.loc[mask, 'viral_activity_level_cat'] = sdf.loc[mask, 'viral_activity_level'].apply(
                lambda x: ViralActivityUtils.get_levels(x, typ='cat'))

        sdf = sdf.rename(columns={'EW_stdate': 'time_value'})
        sdf['viral_activity_level_num'] = sdf['viral_activity_level_cat'].map(ViralActivityUtils.map_level)
        return sdf

class ActivityPipelineRunner:
    def __init__(self, raw_data_path, sewershed_list_path, output_dir, merged_output_dir):
        self.raw_data_path = raw_data_path
        self.sewershed_list_path = sewershed_list_path
        self.output_dir = output_dir
        self.merged_output_dir = merged_output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.merged_output_dir, exist_ok=True)

    def load_data(self):
        return pd.read_csv(self.raw_data_path, parse_dates=['EW_stdate'])

    def load_sewershed_list(self):
        with open(self.sewershed_list_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def process_sewersheds(self, rmdf, ss_list):
        for geo in ss_list:
            sdf = rmdf[rmdf.geo_value == geo]
            processed_df = WastewaterProcessor(rmdf).get_VAL(sdf)
            output_path = os.path.join(self.output_dir, f'{geo}.csv')
            processed_df.to_csv(output_path, index=False)

    def merge_all_sewersheds(self):
        allsdf = pd.DataFrame()
        for filename in os.listdir(self.output_dir):
            try:
                file_path = os.path.join(self.output_dir, filename)
                temp_df = pd.read_csv(file_path, parse_dates=True)
                allsdf = pd.concat([allsdf, temp_df])
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        merged_csv = os.path.join(self.merged_output_dir, 'sewersheds_viral_activity_level.csv')
        allsdf.to_csv(merged_csv, index=False)
        return allsdf

    def compute_region_level(self, df):
        rdf = df.groupby(['region', 'time_value'], as_index=False).median(numeric_only=True)
        rdf['viral_activity_level_num'] = rdf.viral_activity_level.apply(lambda x: ViralActivityUtils.get_levels(x, typ='cat_val'))
        rdf['viral_activity_level_cat'] = rdf.viral_activity_level.apply(lambda x: ViralActivityUtils.get_levels(x, typ='cat'))
        rdf = rdf.drop(columns=['log_viral_load', 'b', 'sd', 'eta'], errors='ignore')
        rdf = rdf.rename(columns={'region': 'geo_value'})
        rdf['geo_res'] = 'region'
        region_csv = os.path.join(self.merged_output_dir, 'region_viral_activity_level.csv')
        rdf.to_csv(region_csv, index=False)

    def compute_state_level(self, df):
        state_df = df.groupby(['time_value'], as_index=False).median(numeric_only=True)
        state_df['viral_activity_level_num'] = state_df.viral_activity_level.apply(lambda x: ViralActivityUtils.get_levels(x, typ='cat_val'))
        state_df['viral_activity_level_cat'] = state_df.viral_activity_level.apply(lambda x: ViralActivityUtils.get_levels(x, typ='cat'))
        state_df = state_df.drop(columns=['log_viral_load', 'b', 'sd', 'eta'], errors='ignore')
        state_df['geo_value'] = 'Virginia'
        state_df['geo_res'] = 'state'
        state_csv = os.path.join(self.merged_output_dir, 'state_viral_activity_level.csv')
        state_df.to_csv(state_csv, index=False)

    def run(self):
        df = self.load_data()
        processor = WastewaterProcessor(df)
        cleaned_df = processor.check_missing_dates_freq()
        sewershed_list = self.load_sewershed_list()
        self.process_sewersheds(cleaned_df, sewershed_list)
        merged_df = self.merge_all_sewersheds()
        self.compute_region_level(merged_df)
        self.compute_state_level(merged_df)
        
class ViralActivityProcessor:
    def __init__(self, data):
        self.data = data

    def get_filtered_data(self, var, geo):
        filtered = self.data[self.data['geo_value'] == geo]
        filtered = filtered[['geo_value', 'time_value', var]]
        return filtered[filtered['time_value'] > '2022-10-01']

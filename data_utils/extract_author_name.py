#!/usr/bin/env python
# coding: utf-8



import pandas as pd

# =============================================================================
# CONFIGURATION — update these paths for your environment
# =============================================================================
FINAL_DATA_CSV       = r"PATH_TO_FINAL_DATA_CSV"        # e.g. data/final_data.csv
DATA_23_CSV          = r"PATH_TO_DATA_23_CSV"            # e.g. data/data_23.csv
OUTPUT_CSV           = r"PATH_TO_OUTPUT_CSV"             # e.g. data/final_data_with_author_names.csv
# =============================================================================

d1 = pd.read_csv(FINAL_DATA_CSV)
d2 = pd.read_csv(DATA_23_CSV)



d1




print(d1.columns)
print(d2.columns)



d2



# Ensure keys are the same type
d1['ha_id'] = d1['ha_id'].astype(str)
d2['author_id'] = d2['author_id'].astype(str)

# Create lookup dictionaries from d2 for all columns we want to add
author_map = dict(zip(d2['author_id'], d2['author_name']))
time_map = dict(zip(d2['author_id'], d2['time']))
weekday_map = dict(zip(d2['author_id'], d2['weekday']))
month_map = dict(zip(d2['author_id'], d2['month']))
year_map = dict(zip(d2['author_id'], d2['year']))

# Map all columns to d1
d1['author_name'] = d1['ha_id'].map(author_map)
d1['time'] = d1['ha_id'].map(time_map)
d1['weekday'] = d1['ha_id'].map(weekday_map)
d1['month'] = d1['ha_id'].map(month_map)
d1['year'] = d1['ha_id'].map(year_map)



d1



d1.to_csv(OUTPUT_CSV, index=False)





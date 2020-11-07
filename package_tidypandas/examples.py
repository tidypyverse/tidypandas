import pandas as pd
bmt = pd.read_feather("work/ltv_modeling/data/training_datasets/dataset_2019-01-01.feather")
bmt
bmt.info()

bmt_tidy = tidy(bmt)
bmt_tidy
bmt_tidy.info()

bmt_tidy.group_by(['n_rides'])

bmt_tidy_2 = bmt_tidy.select(['user_id', 'n_rides'])
bmt_tidy_2
bmt_tidy

bmt_tidy_3 = bmt_tidy.select(['user_id', 'n_rides'], include = False)
bmt_tidy_3
bmt_tidy

bmt_tidy_grouped = bmt_tidy.group_by(['n_rides'])
bmt_tidy_grouped.info()
bmt_tidy_grouped.select(['user_id'])

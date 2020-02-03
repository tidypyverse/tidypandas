# tidypandas

Status: WIP

 - tidyverse-like utils for working pandas dataframes
 - Use with `pd.pipe`

| Function | Input | Output |
| -------- | ----- | ------ |
| `tidy_count` | grouped dataframe | ungrouped dataframe with count column(default name is 'n') added to groupby columns |
| `tidy_add_count` | grouped dataframe | ungrouped dataframe with count column(default name is 'n') added to input dataframe |
| `tidy_ungroup`   | grouped dataframe | ungrouped dataframe |

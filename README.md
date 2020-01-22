# tidypandas

Status: WIP

 - dplyr-like utils for working pandas dataframes
 - Use with `pd.pipe`

| Function | Input | Output |
| -------- | ----- | ------ |
| `dplyr_count` | grouped dataframe | ungrouped dataframe with count column(default name is 'n') added to groupby columns |
| `dplyr_add_count` | grouped dataframe | ungrouped dataframe with count column(default name is 'n') added to input dataframe |
| `dplyr_ungroup`   | grouped dataframe | ungrouped dataframe |

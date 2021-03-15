# tidypandas
`make pandas talk dplyr`

**Under Construction**

Implementation notes (for the developer team):

## Basic Verbs (one-table)

- [x] select
- [x] group_by
- [x] ungroup
- [x] mutate
- [x] arrange
- [x] filter
- [x] distinct
- [x] summarise
- [x] slice

## class methods

- [x] init
- [x] repr
- [x] to_pandas
- [x] get_nrow
- [x] get_ncol
- [x] get_colnames
- [x] get_shape
- [x] get_dim
- [x] get_groupvars for tidy grouped
- [x] to_dict for tidy grouped
- [x] pipe
- [x] pipe2 (or pipe_pandas)
- [x] to_series

## join methods

- [x] join method for tidy (inner, outer, left, right, anti are special cases)
- [x] join method for tidy grouped

## bind methods

- [x] cbind
- [x] rbind

## util methods

- [x] count
- [x] add_count
 
## pivot methods

- [x] pivot_longer
- [x] pivot_wider

## slice methods

- [x] slice_head
- [x] slice_tail
- [x] slice_sample
- [x] slice_bootstrap
- [x] slice_min
- [x] slice_max

## apply methods

- [x] group_modify

## set-like methods (for ungrouped only)

- [x] union
- [x] intersect
- [x] setdiff

## completing methods

- [ ] expand
- [ ] complete

## na methods

- [x] fill_na
- [x] replace_na
- [x] drop_na

## character column helper methods

- [x] separate
- [x] unite
- [x] separate_rows

## functions

- [x] bind_rows
- [x] bind_cols



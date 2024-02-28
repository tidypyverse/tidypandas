# Changelog

## v0.3.1 (24th Jan 2024)
- Handled deprecation [warnings](https://github.com/tidypyverse/tidypandas/issues/51) created by pandas >= 2.0.0. Use of `group_modify` still produces warning. This is doe to the fact that `include_groups` argument needs to be set to False by default.
- `unnest` method allows a list of numpy ndarrays (as column to be unnested)
- `unnest_wider` method is introduced
- `fill_na` reimplemented to be performant (using Series.groupby.xfill), no user facing change


## v0.3.0 (16th Aug 2023)
- `tidyselect` is introduced in a few verbs (methods) which support `start_with`, `ends_with` and `contains`.
- `summarise` now supports returning an iterable within a list as a output, no longer restricts the output to a scalar.

## v0.2.6 (18th June 2023)
- pivot_wider gains a new argument `id_expand` (False by default)

## v0.2.5 (13th June 2023)
- slice_max and slice_min are rewritten to avoid pandas apply and using pandas.groupbBy.Series.rank, thereby being fast for larger datasets.

## v0.2.4 (27th Jan 2023)
- tidyframe drops row indexes when constructed using a pandas dataframe
- User is expected to run tidypandas.tidy_utils.simplify() beforehand to
  handle complicated pandas dataframe input

## v0.2.3 (19th Oct 2022)
- bugfix: show proper error message when tidyframe cannot be created from pandas dataframe with some column starting with underscore
- bugfix: tidyframe.separate now handles NAs

## v0.2.2 (28 Jun 2022)
- minor bug in _is_kwargable is fixed, making mutate better 
- minor bug in filter with mask is fixed
- CI is enabled with github actions

## v0.2.1 (14th June 2022)
- glimpse by default shows 100 columns
- add_row_number is more efficient

## v0.2.0 (12th June 2022)
- show method added (print specified number of rows)
- glimpse method added (to see dataframe in horizontal form)
- add_row_number is fast in grouped case (use of .cumcount)
- All changes of github only release v0.1.5

## v0.2.0 (12th June 2022)
- show method added (print specified number of rows)
- glimpse method added (to see dataframe in horizontal form)
- add_row_number is fast in grouped case (use of .cumcount)
- All changes of github only release v0.1.5

## v0.1.5 (7th June 2022)
- methods 'expand' and 'complete' are implemented
- 'rename' method gains arguments 'predicate' and 'func'
- 'pivot_wider' method gains argument 'names_prefix'
- 'skim' method now provides a warning when skimpy is not installed
- utility 'expand_grid' is implemented

## v0.1.4 (14/05/2022)
- __init__ now does not coerce column classes

## v0.1.3 (27/04/2022)
- __repr__ set right for dataframes with pandas version <= 1.4.0
- tidyframe __init__ now accepts inputs and passes them to pd.DataFrame
- Along with accepting a pandas dataframe

## v0.1.2 (24/04/2022)
- minor bugfix where `skimpy` is now optional!

## v0.1.1 (22/04/2022)
- Patch for First release of `tidypandas`!

## v0.1.0 (22/04/2022)
- First release of `tidypandas`!

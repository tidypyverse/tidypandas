![](docs/logo.png)

[![PyPI
version](https://badge.fury.io/py/tidypandas.svg)](https://badge.fury.io/py/tidypandas)

# `tidypandas`

> A **grammar of data manipulation** for
> [pandas](https://pandas.pydata.org/docs/index.html) inspired by
> [tidyverse](https://tidyverse.tidyverse.org/)

`tidypandas` python package provides *minimal, pythonic* API for common
data manipulation tasks:

-   `tidyframe` class (wrapper over pandas dataframe) provides a
    dataframe with simplified index structure (no more resetting indexes
    and multi indexes)
-   Consistent ‘verbs’ (`select`, `arrange`, `distinct`, …) as methods
    to `tidyframe` class which mostly return a `tidyframe`
-   Unified interface for summarizing (aggregation) and mutate (assign)
    operations across groups
-   Utilites for pandas dataframes and series
-   Uses simple python data structures, No esoteric classes, No pipes,
    No Non-standard evaluation
-   No copy data conversion between `tidyframe` and pandas dataframes
-   An accessor to apply `tidyframe` verbs to simple pandas datarames
-   …

## Example

-   `tidypandas` code:

<!-- -->

    df.filter(lambda x: x['col_1'] > x['col_1'].mean(), by = 'col_2')

-   equivalent pandas code:

<!-- -->

    (df.groupby('col2')
       .apply(lambda x: x.loc[x['col_1'] > x['col_1'].mean(), :])
       .reset_index(drop = True)
       )

## Why use `tidypandas`

`tidypandas` is for you if:

-   you *frequently* write data manipulation code using pandas
-   you prefer to have stay in pandas ecosystem (see accessor)
-   you *prefer* to remember a [limited set of
    methods](https://medium.com/dunder-data/minimally-sufficient-pandas-a8e67f2a2428)
-   you do not want to write (or be surprised by)
    [`reset_index`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html),
    [`rename_axis`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename_axis.html)
    often
-   you prefer writing free flowing, expressive code in
    [dplyr](https://dplyr.tidyverse.org/) style

> `tidypandas` relies on the amazing `pandas` library and offers a
> consistent API with a different
> [philosophy](https://tidyverse.tidyverse.org/articles/manifesto.html).

## Presentation

Learn more about tidypandas
([presentation](https://github.com/talegari/tidypandas/blob/master/docs/tp_pres.html))

## Installation

1.  Install release version from Pypi using pip:

        pip install tidypandas

2.  For offline installation, use whl/tar file from the [releases
    page](https://github.com/talegari/tidypandas/releases) on github.

## Contribution/bug fixes/Issues:

1.  Open an issue/suggestion/bugfix on the github
    [issues](https://github.com/talegari/tidypandas/issues) page.

2.  Use the master branch from
    [github](https://github.com/talegari/tidypandas) repo to submit your
    PR.

------------------------------------------------------------------------

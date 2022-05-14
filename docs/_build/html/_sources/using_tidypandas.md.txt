# Using `tidypandas`

> [`tidypandas`](https://github.com/talegari/tidypandas) – A **grammar
> of data manipulation** for
> [pandas](https://pandas.pydata.org/docs/index.html) inspired by
> [tidyverse](https://tidyverse.tidyverse.org/)

## Overview of tidypandas

A pandas dataframe is said to be ‘simple’ if:

1.  Column names (x.columns) are an unnamed pd.Index object of unique
    strings.
2.  Row names (x.index) are an unnamed pd.RangeIndex object with start =
    0 and step = 1.

`tidypandas` provides the following utilities:

1.  `tidyframe` class: A wrapper over a ‘simple’ pandas datadrame with
    verbs as methods.
2.  `tidy_utils`: Functions `simplify`, `is_simple` to simplify as
    pandas dataframe.
3.  `series_utils`: Functions like `coalesce`, `case_when`, `min_rank`.
4.  `tp` accessor: Use `tidypandas` verbs directly on pandas dataframes
    with pandas dataframes as output.

## Installation

    pip install tidypandas

## Imports

    from tidypandas import tidyframe
    from tidypandas.tidy_utils import simplify
    from tidypandas.series_utils import *
    from tidypandas.tidy_accessor import tp

## Creating a `tidyframe`

A `tidyframe` is created from an existing pandas dataframe (by default
makes a copy).

    from palmerpenguins import load_penguins
    penguins      = load_penguins() # pandas dataframe
    penguins_tidy = tidyframe(penguins) # create a tidyframe from pandas dataframe
    print(penguins_tidy)

    ## # A tidy dataframe: 344 X 8
    ##    species     island  bill_length_mm  ...  body_mass_g      sex    year
    ##   <object>   <object>       <float64>  ...    <float64> <object> <int64>
    ## 0   Adelie  Torgersen            39.1  ...       3750.0     male    2007
    ## 1   Adelie  Torgersen            39.5  ...       3800.0   female    2007
    ## 2   Adelie  Torgersen            40.3  ...       3250.0   female    2007
    ## 3   Adelie  Torgersen             NaN  ...          NaN      NaN    2007
    ## 4   Adelie  Torgersen            36.7  ...       3450.0   female    2007
    ## 5   Adelie  Torgersen            39.3  ...       3650.0     male    2007
    ## 6   Adelie  Torgersen            38.9  ...       3625.0   female    2007
    ## 7   Adelie  Torgersen            39.2  ...       4675.0     male    2007
    ## 8   Adelie  Torgersen            34.1  ...       3475.0      NaN    2007
    ## 9   Adelie  Torgersen            42.0            4250.0      NaN    2007
    ## #... with 334 more rows

## Working with `tidyframe`s

The methods of `tidyframe` class are ‘verbs’ like:

-   `select` (subset some columns)
-   `filter` (subset some rows based on conditions)
-   `arrange` (order the rows)
-   `slice` (subset some rows)
-   `distinct` (subset rows by distinct values of one or more columns)
-   `mutate` (add or modify an existing column)
-   `summarise` (aggregate some columns)

and many more.

Typically, a method call on a `tidyframe` object returns a new
`tidyframe` object. Only `[` (`setitem`) method makes assignment
in-place.

An operation on a `tidyframe`(s) can be achieved by composition of
methods or verbs.

> example: Obtain count of birds per specie in the ‘Dream’ island

    print( penguins_tidy.filter("island == 'Dream'").count('species') )

    ## # A tidy dataframe: 2 X 2
    ##      species       n
    ##     <object> <int64>
    ## 0  Chinstrap      68
    ## 1     Adelie      56

## Exporting a `tidyframe` to pandas

A `tidyframe` can be exported as a pandas dataframe using `to_pandas`
method.

## Using accessor

`tidypandas` provides the ability to use the ‘verbs’ directly on
‘simple’ pandas dataframes and get the result back as a pandas
dataframe. The methods should be prepended by `tp` (short for
`tidypandas`).

    penguins.tp.slice([0, 1], by = 'species')

    ##      species     island  bill_length_mm  ...  body_mass_g     sex  year
    ## 0     Adelie  Torgersen            39.1  ...       3750.0    male  2007
    ## 1     Adelie  Torgersen            39.5  ...       3800.0  female  2007
    ## 2     Gentoo     Biscoe            46.1  ...       4500.0  female  2007
    ## 3     Gentoo     Biscoe            50.0  ...       5700.0    male  2007
    ## 4  Chinstrap      Dream            46.5  ...       3500.0  female  2007
    ## 5  Chinstrap      Dream            50.0  ...       3900.0    male  2007
    ## 
    ## [6 rows x 8 columns]

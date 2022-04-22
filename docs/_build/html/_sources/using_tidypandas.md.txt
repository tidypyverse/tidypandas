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
    penguins_tidy

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
<tr>
<td style="color:#606060;font-style:oblique;"></td>
<td style="color:#606060;font-style:oblique;">&lt;string&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;string&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;Float64&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;Float64&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;Int64&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;Int64&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;string&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;Int64&gt;</td>
</tr>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181</td>
      <td>3750</td>
      <td>male</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186</td>
      <td>3800</td>
      <td>female</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195</td>
      <td>3250</td>
      <td>female</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193</td>
      <td>3450</td>
      <td>female</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>339</th>
      <td>Chinstrap</td>
      <td>Dream</td>
      <td>55.8</td>
      <td>19.8</td>
      <td>207</td>
      <td>4000</td>
      <td>male</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>340</th>
      <td>Chinstrap</td>
      <td>Dream</td>
      <td>43.5</td>
      <td>18.1</td>
      <td>202</td>
      <td>3400</td>
      <td>female</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Chinstrap</td>
      <td>Dream</td>
      <td>49.6</td>
      <td>18.2</td>
      <td>193</td>
      <td>3775</td>
      <td>male</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Chinstrap</td>
      <td>Dream</td>
      <td>50.8</td>
      <td>19.0</td>
      <td>210</td>
      <td>4100</td>
      <td>male</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>343</th>
      <td>Chinstrap</td>
      <td>Dream</td>
      <td>50.2</td>
      <td>18.7</td>
      <td>198</td>
      <td>3775</td>
      <td>female</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
<p>344 rows × 8 columns</p>
</div>

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

    penguins_tidy.filter("island == 'Dream'").count('species')

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
<tr>
<td style="color:#606060;font-style:oblique;"></td>
<td style="color:#606060;font-style:oblique;">&lt;string&gt;</td>
<td style="color:#606060;font-style:oblique;">&lt;int64&gt;</td>
</tr>
    <tr>
      <th>0</th>
      <td>Chinstrap</td>
      <td>68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
</div>

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
    ## 0     Adelie  Torgersen            39.1  ...         3750    male  2007
    ## 1     Adelie  Torgersen            39.5  ...         3800  female  2007
    ## 2     Gentoo     Biscoe            46.1  ...         4500  female  2007
    ## 3     Gentoo     Biscoe            50.0  ...         5700    male  2007
    ## 4  Chinstrap      Dream            46.5  ...         3500  female  2007
    ## 5  Chinstrap      Dream            50.0  ...         3900    male  2007
    ## 
    ## [6 rows x 8 columns]

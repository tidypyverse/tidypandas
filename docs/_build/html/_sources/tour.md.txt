# A tour of `tidypandas`

------------------------------------------------------------------------

> The intent of this document is to illustrate some standard data
> manipulation exercises using
> [`tidypandas`](https://github.com/talegari/tidypandas) python package.
> We use the `nycflights13` data.

------------------------------------------------------------------------

> [`tidypandas`](https://github.com/talegari/tidypandas) – A **grammar
> of data manipulation** for
> [pandas](https://pandas.pydata.org/docs/index.html) inspired by
> [tidyverse](https://tidyverse.tidyverse.org/)

------------------------------------------------------------------------

`nycflights13` contains information about all flights that departed from
NYC (e.g. EWR, JFK and LGA) to destinations in the United States, Puerto
Rico, and the American Virgin Islands) in 2013: 336,776 flights in
total. To help understand what causes delays, it also includes a number
of other useful datasets.

-   flights: all flights that departed from NYC in 2013
-   weather: hourly meteorological data for each airport
-   planes: construction information about each plane
-   airports: airport names and locations
-   airlines: translation between two letter carrier codes and names

## Imports

    from tidypandas import tidyframe
    from tidypandas.series_utils import *
    import plotnine as gg

## Load and Display `flights` data

    from nycflights13 import flights, planes
    flights_tidy = tidyframe(flights)
    print(flights_tidy)

    ## # A tidy dataframe: 336776 X 19
    ##      year   month     day  dep_time  sched_dep_time  dep_delay  arr_time  ...
    ##   <int64> <int64> <int64> <float64>         <int64>  <float64> <float64>  ...
    ## 0    2013       1       1     517.0             515        2.0     830.0  ...
    ## 1    2013       1       1     533.0             529        4.0     850.0  ...
    ## 2    2013       1       1     542.0             540        2.0     923.0  ...
    ## 3    2013       1       1     544.0             545       -1.0    1004.0  ...
    ## 4    2013       1       1     554.0             600       -6.0     812.0  ...
    ## 5    2013       1       1     554.0             558       -4.0     740.0  ...
    ## 6    2013       1       1     555.0             600       -5.0     913.0  ...
    ## 7    2013       1       1     557.0             600       -3.0     709.0  ...
    ## 8    2013       1       1     557.0             600       -3.0     838.0  ...
    ## 9    2013       1       1     558.0             600       -2.0     753.0     
    ## #... with 336766 more rows, and 12 more columns: sched_arr_time <int64>, arr_delay <float64>, carrier <object>, flight <int64>, tailnum <object>, origin <object>, dest <object>, air_time <float64>, distance <int64>, hour <int64>, minute <int64>, time_hour <object>

## Exercise: Find all flights that arrived more than two hours late, but didn’t leave late.

    out = (flights_tidy.filter("dep_delay <= 0 and arr_delay > 120")
                       .select(['flight', 'dep_delay', 'arr_delay'])
                       )
    print(out)

    ## # A tidy dataframe: 29 X 3
    ##    flight  dep_delay  arr_delay
    ##   <int64>  <float64>  <float64>
    ## 0    3728       -1.0      124.0
    ## 1    5181        0.0      130.0
    ## 2    1151       -2.0      124.0
    ## 3       3       -3.0      122.0
    ## 4     399       -2.0      194.0
    ## 5     389       -3.0      140.0
    ## 6    4540       -5.0      124.0
    ## 7     707       -2.0      179.0
    ## 8    2083       -5.0      143.0
    ## 9    4674       -3.0      127.0
    ## #... with 19 more rows

## Exercise: Sort flights to find the fastest flights

    out = (flights_tidy.mutate({'speed': (lambda x, y: x/y, ['distance', 'air_time'])})
                       .arrange([('speed', 'desc')])
                       .select(['flight', 'dep_delay', 'arr_delay', 'speed'])
                       )
    print(out)

    ## # A tidy dataframe: 336776 X 4
    ##    flight  dep_delay  arr_delay      speed
    ##   <int64>  <float64>  <float64>  <float64>
    ## 0    1499        9.0      -14.0  11.723077
    ## 1    4667       45.0       26.0  10.838710
    ## 2    4292       15.0       -1.0  10.800000
    ## 3    3805        4.0        2.0  10.685714
    ## 4    1902       -1.0      -28.0   9.857143
    ## 5     315       -5.0      -51.0   9.400000
    ## 6     707       -3.0      -26.0   9.290698
    ## 7     936       -1.0      -43.0   9.274286
    ## 8     347        1.0      -32.0   9.236994
    ## 9     329       -2.0      -39.0   9.236994
    ## #... with 336766 more rows

## Exercise: Is the proportion of cancelled flights related to the average delay?

    out = (flights_tidy
            .mutate({'cancelled': (lambda x, y: (pd.isna(x) | pd.isna(y)),
                                   ['arr_delay', 'dep_delay']
                                  )
                    }
                   )
            .summarise({'cancelled_prop': (np.mean, 'cancelled'),
                        'avg_dep_delay': (np.mean, 'dep_delay'),
                        'avg_arr_delay': (np.mean, 'arr_delay')
                       },
                       by = ['year', 'month', 'day']
                      )
            )
                            
    print(out)

    ## # A tidy dataframe: 365 X 6
    ##      year   month     day  cancelled_prop  avg_dep_delay  avg_arr_delay
    ##   <int64> <int64> <int64>       <float64>      <float64>      <float64>
    ## 0    2013       1       1        0.013064      11.548926      12.651023
    ## 1    2013       1       2        0.015907      13.858824      12.692888
    ## 2    2013       1       3        0.015317      10.987832       5.733333
    ## 3    2013       1       4        0.007650       8.951595      -1.932819
    ## 4    2013       1       5        0.004167       5.732218      -1.525802
    ## 5    2013       1       6        0.003606       7.148014       4.236429
    ## 6    2013       1       7        0.003215       5.417204      -4.947312
    ## 7    2013       1       8        0.007786       2.553073      -3.227578
    ## 8    2013       1       9        0.009978       2.276477      -0.264278
    ## 9    2013       1      10        0.003219       2.844995      -5.898816
    ## #... with 355 more rows

    data_for_plot = (out.pivot_longer(cols = ['avg_dep_delay', 'avg_arr_delay'],
                                      names_to = "delay_type"
                                      )
                        )
    print(data_for_plot)

    ## # A tidy dataframe: 730 X 6
    ##       day  cancelled_prop   month    year     delay_type      value
    ##   <int64>       <float64> <int64> <int64>       <object>  <float64>
    ## 0       1        0.013064       1    2013  avg_dep_delay  11.548926
    ## 1       2        0.015907       1    2013  avg_dep_delay  13.858824
    ## 2       3        0.015317       1    2013  avg_dep_delay  10.987832
    ## 3       4        0.007650       1    2013  avg_dep_delay   8.951595
    ## 4       5        0.004167       1    2013  avg_dep_delay   5.732218
    ## 5       6        0.003606       1    2013  avg_dep_delay   7.148014
    ## 6       7        0.003215       1    2013  avg_dep_delay   5.417204
    ## 7       8        0.007786       1    2013  avg_dep_delay   2.553073
    ## 8       9        0.009978       1    2013  avg_dep_delay   2.276477
    ## 9      10        0.003219       1    2013  avg_dep_delay   2.844995
    ## #... with 720 more rows

    (gg.ggplot(data_for_plot.to_pandas(),
            gg.aes('value', 'cancelled_prop', color = 'delay_type')
            ) +
        gg.geom_point() +
        gg.geom_smooth(method = "lm")
        )

    ## <ggplot: (305770403)>

<img src="tour_files/figure-markdown_strict/unnamed-chunk-6-1.png" width="614" />

## Exercise: Find all destinations that are flown by at least two carriers. Use that information to rank the carriers.

    out = (flights_tidy.mutate({'n_carriers': (n_distinct, 'carrier')}, by = 'dest')
                       .filter('n_carriers > 1')
                       .summarise({'n_dest': (n_distinct, 'dest')}, by = 'carrier')
                       .arrange([('n_dest', 'desc')])
                       )
    print(out)

    ## # A tidy dataframe: 16 X 2
    ##    carrier  n_dest
    ##   <object> <int64>
    ## 0       EV      51
    ## 1       9E      48
    ## 2       UA      42
    ## 3       DL      39
    ## 4       B6      35
    ## 5       AA      19
    ## 6       MQ      19
    ## 7       WN      10
    ## 8       US       5
    ## 9       OO       5
    ## #... with 6 more rows

## Exercise: Is there a relationship between the age of a plane and its delays?

    planes_tidy = tidyframe(planes)
    print(planes_tidy)

    ## # A tidy dataframe: 3322 X 9
    ##    tailnum      year                     type      manufacturer      model  engines  ...
    ##   <object> <float64>                 <object>          <object>   <object>  <int64>  ...
    ## 0   N10156    2004.0  Fixed wing multi engine           EMBRAER  EMB-145XR        2  ...
    ## 1   N102UW    1998.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2  ...
    ## 2   N103US    1999.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2  ...
    ## 3   N104UW    1999.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2  ...
    ## 4   N10575    2002.0  Fixed wing multi engine           EMBRAER  EMB-145LR        2  ...
    ## 5   N105UW    1999.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2  ...
    ## 6   N107US    1999.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2  ...
    ## 7   N108UW    1999.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2  ...
    ## 8   N109UW    1999.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2  ...
    ## 9   N110UW    1999.0  Fixed wing multi engine  AIRBUS INDUSTRIE   A320-214        2     
    ## #... with 3312 more rows, and 3 more columns: seats <int64>, speed <float64>, engine <object>

    planes_year_frame = (planes_tidy.select(['tailnum', 'year'])
                                    .rename({'year': 'plane_year'})
                                    )
    print(planes_year_frame)

    ## # A tidy dataframe: 3322 X 2
    ##    tailnum  plane_year
    ##   <object>   <float64>
    ## 0   N10156      2004.0
    ## 1   N102UW      1998.0
    ## 2   N103US      1999.0
    ## 3   N104UW      1999.0
    ## 4   N10575      2002.0
    ## 5   N105UW      1999.0
    ## 6   N107US      1999.0
    ## 7   N108UW      1999.0
    ## 8   N109UW      1999.0
    ## 9   N110UW      1999.0
    ## #... with 3312 more rows

    age_delay_stats_frame = \
      (flights_tidy.inner_join(planes_year_frame, on = 'tailnum')
                   .mutate({'age': (lambda x, y: x - y, ['year', 'plane_year'])})
                   .filter(lambda x: ~ pd.isna(x['age']))
                   .mutate({'age_25': lambda x: ifelse(x['age'] > 25, 25, x['age'])})
                   .summarise(column_names = ['arr_delay', 'dep_delay'],
                              func = np.mean,
                              prefix = 'mean_',
                              by = 'age'
                              )
                   )
    print(age_delay_stats_frame)

    ## # A tidy dataframe: 46 X 3
    ##         age  mean_arr_delay  mean_dep_delay
    ##   <float64>       <float64>       <float64>
    ## 0      14.0        7.117146       13.079679
    ## 1      15.0        6.709817       13.429565
    ## 2      23.0        5.694890       11.504746
    ## 3       1.0        2.850889        9.642778
    ## 4      22.0        4.177950       10.674242
    ## 5      13.0        7.259624       11.734242
    ## 6       9.0       10.243436       16.387705
    ## 7       2.0        5.696238       11.840951
    ## 8       6.0        7.540417       13.737950
    ## 9       5.0        5.572951       13.158852
    ## #... with 36 more rows

    (gg.ggplot(age_delay_stats_frame.to_pandas(),
               gg.aes('age', 'mean_arr_delay')
               ) +
         gg.geom_point() +
         gg.xlim(0, 20) +
         gg.ylim(0, 11)
         )

    ## <ggplot: (307959763)>
    ## 
    ## /Users/s0k06e8/tpenv/lib/python3.9/site-packages/plotnine/layer.py:401: PlotnineWarning: geom_point : Removed 25 rows containing missing values.

<img src="tour_files/figure-markdown_strict/unnamed-chunk-8-3.png" width="614" />

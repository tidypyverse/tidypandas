# to serve as a tutorial for tidypandas
from nycflights13 import *

# we have five pandas dataframes
flights
weather
planes
airports
airlines

flights_tidy  = tidyDataFrame(flights)
weather_tidy  = tidyDataFrame(weather)
planes_tidy   = tidyDataFrame(planes)
airports_tidy = tidyDataFrame(airports)
airlines_tidy = tidyDataFrame(flights)

flights_tidy.get_info()

# what characterizes a tidy dataframe

# - column names are strings, and form unique set of names (no multiindex, no name)
# - row index is absent (no index or miltiindex)
# - Most importantly, method calls on a tidydataframe results in a tidydataframe again (most of the times except for methods like to_pandas)

# Package defines two classes: tidyDataFrame and tidyGroupedDataFrame.

# 1
# Suppose a day-month combination defines a group.
# keep three rows from each group with the top dep_delay
# display these columns: year, day, month, dep_delay, time_hour, arr_delay

# tidypandas
(flights_tidy.group_by(['day', 'month'])
             .slice_max(n = 3, order_by = 'dep_delay')
             .select(['year', 'day', 'month', 'dep_delay'
                      , 'time_hour', 'arr_delay'])
             )

# pandas
(flights.groupby(['day', 'month'])
        .apply(lambda x: x.nlargest(n = 3
                                    , columns = 'dep_delay'
                                    , keep = "all"
                                    )
               )
        .reset_index(drop = True)
        .loc[:, ['year', 'day', 'month', 'dep_delay'
                      , 'time_hour', 'arr_delay']]
        .groupby(['day', 'month'])
        )

temp = (flights.groupby(['day', 'month'])
             .apply(lambda x: x.nlargest(n = 3
                                         , columns = 'dep_delay'
                                         , keep = "all"
                                         )
                    )
             )
tidy(temp)

# if you did not want to make this computation per group, then code would not change much:

# tidypandas
(flights_tidy.slice_max(n = 3, order_by = 'dep_delay')
             .select(['year', 'day', 'month', 'dep_delay'
                      , 'time_hour', 'arr_delay'])
             )

# pandas
(flights.nlargest(n = 3, columns = 'dep_delay', keep = "all")
        .loc[:, ['year', 'day', 'month', 'dep_delay'
                      , 'time_hour', 'arr_delay']]
        )

# 2
# popular trips are ones which have atleast one flight everyday of the year

# tidypandas
(flights_tidy.distinct(['origin', 'dest', 'month', 'day'])
             .count(['origin', 'dest'])
             .filter('n == 365')
             .select(['origin', 'dest'])
             )


# pandas
(flights.drop_duplicates(['origin', 'dest', 'month', 'day'])
        .value_counts(['origin', 'dest']) # this is a series at this stage
        .reset_index()
        .rename(columns = {0 : 'count'})
        .query('count == 365')
        .loc[:, ['origin', 'dest']]
        )

# 3
# simple crosstab implementation

temp = pd.crosstab(flights['origin'], flights['dest']).reset_index(drop = False)
temp2 = copy.copy(temp)
temp.columns.name = None
temp

tidy(temp2)

def crosstab_tidy(df, two_string_list):
    
    assert isinstance(two_string_list, list)
    assert len(two_string_list) == 2
    assert all([isinstance(x, str) for x in two_string_list])
    
    res = (df.count(two_string_list, count_column_name = "__count")
             .pivot_wider(id_cols       = two_string_list[0]
                          , names_from  = two_string_list[1]
                          , values_from = '__count'
                          , values_fill = 0
                          )
             )

    return res

crosstab_tidy(flights_tidy, ['origin', 'dest'])

# 4
# grouping is saved unless there is a summarization or an ungroup call

# tidypandas
flights_tidy.group_by('dest').slice_sample(prop = 0.1)

# pandas
flights.groupby('dest').sample(frac = 0.1)
tidyDataFrame(flights.groupby('dest').sample(frac = 0.1))
tidy(flights.groupby('dest').sample(frac = 0.1))

# 5
# mutate and summarize cover multiple paradigms underneath

numeric_dtypes = [np.float64, np.float32,np.int32, np.int64]
flights_tidy.mutate(predicate = lambda x: x.dtypes in numeric_dtypes,
                    func = lambda x: (x - x.min())/(x.max() - x.min())
                    )
(flights
    .select_dtypes(include = numeric_dtypes)
    .apply(lambda x: ((x - x.min())/(x.max() - x.min())), axis = "index")
    .join(flights.select_dtypes(exclude= numeric_dtypes))
    )

iris_tidy = tidyDataFrame(pd.read_csv("~/tidypandas/iris.csv"))

# 6
# all styles can be used within a single call
# mutate executes in order
iris_tidy.mutate({"sl"           : lambda x : x['Sepal.Length'] + x.shape[0],
                  "pl"           : (lambda x: x + 1, ('Petal.Length')),
                  "Petal.Length" : (lambda x: x + 2, )
                 }
                )

(iris_tidy.group_by('Species')
          .mutate({"sl"           : lambda x : x['Sepal.Length'] + x.shape[0],
                   "pl"           : (lambda x: x + 1, 'Petal.Length'),
                   "Petal.Length" : (lambda x: x + 2, )
                  }
                 )
          )

# 7
# which carrier shows up most often in flights table but not found in planes

# tidypandas
(flights_tidy.distinct(["carrier", "tailnum"])
             .join_left(planes_tidy, on = "tailnum")
             .group_by('carrier')
             .summarise({'total_planes'  : (lambda x: len(x), 'carrier'),
                         'not_in_planes' : (lambda x: x.isna().sum(), 'model')
                        }
                       )
             .mutate({'missing_pct' : (lambda x, y: x/y, ('not_in_planes', 'total_planes'))})
             .arrange('missing_pct', ascending = False)
             )

# pandas
def count_na(x):
    return x.isna().sum()

# pandas (srikanth)
(flights.loc[:, ["carrier", "tailnum"]]
        .drop_duplicates(["carrier", "tailnum"])
        .merge(planes, on = "tailnum", how = "left")
        .groupby("carrier")
        .apply(lambda chunk: pd.DataFrame({'total_planes' : len(chunk['model']),
                                           'not_in_planes': chunk['model'].isna().sum()
                                           }
                                          , index = [0]
                                          )
               )
        .droplevel(1)
        .reset_index()
        # next line is difficult to program with
        .assign(missing_pct = lambda x: x['not_in_planes']/x['total_planes'])
        .sort_values('missing_pct', ascending = False)
        )

# pandas (ashish)
flights_mod = (flights.loc[:, ["carrier", "tailnum"]]
                    .drop_duplicates(["carrier", "tailnum"])
                    .merge(planes, on = "tailnum", how = "left")
                    .groupby("carrier")
                    .agg({"model": [len, count_na]})
                    .reset_index())
        
flights_mod.columns = ["_".join(filter(lambda x: x!="", c)) if len(c) > 1 else c for c in flights_mod.columns]

flights_mod["missing_pct"] = flights_mod["model_count_na"]/flights_mod["model_len"]
flights_mod = flights_mod.sort_values("missing_pct", ascending=False)
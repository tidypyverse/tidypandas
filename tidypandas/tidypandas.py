import pandas as pd
import warnings
import inspect
import numpy as np
import pandas as pandas
training = pandas.read_csv("work/ltv_modeling/data/training.csv.gz")
training

training.info()
training.describe()

#---------------------------------------------
def is_pandas_frame(df):
    
    res = df.__class__.__name__ in ["DataFrame", "DataFrameGroupBy"]
    
    return(res)

#---------------------------------------------  
def is_grouped_frame(df, check = True):
    
    if check:
        assert is_pandas_frame(df)
    
    res = df.__class__.__name__ == "DataFrameGroupBy"
    return(res)

#---------------------------------------------
def group_vars(df, check = True):
    
    if check:
        assert is_pandas_frame(df)
    
    if is_grouped_frame(df, check = False):
        res = df.grouper.names
    else:
        res = []
    
    return res

#---------------------------------------------
def is_tidy_frame(df):
    
    assert is_pandas_frame(df)
    
    # groupby object specific checks
    res = True
    if is_grouped_frame(df, check = False):
        groupvars = group_vars(df, check = False)
        if len(groupvars) == 0:
            res = False
        if not isinstance(df.grouper.axis, (pd.RangeIndex, pd.Int64Index)):
            res = False
        df = df.obj
    # all groupby object checks are done
    # proceed with DataFrame checks if necessary
    if res:
        # check if row and column indexes are not multiindexes
        row_flag = not isinstance(df.index, pd.MultiIndex)
        columns  = list(df.columns)
        col_flag = not isinstance(columns, pd.MultiIndex)
        # check if row index is rangeIndex
        flag_no_index = isinstance(df.index, (pd.RangeIndex, pd.Int64Index))
        # check if all column names are strings
        str_flag = all(map(lambda x:isinstance(x, str), columns))
        # check if column names are unique
        if len(set(columns)) == len(columns):
            unique_flag = True
        else:
            unique_flag = False
        
        res = (row_flag
              and col_flag 
              and flag_no_index 
              and str_flag 
              and unique_flag)
    
    return(res)

#---------------------------------------------
def ungroup(df, check = True):
    if check:
        assert is_tidy_frame(df)
    
    if is_grouped_frame(df, check = False):
        res = df.obj
    else:
        res = df
    return res

#---------------------------------------------
def count(df, cols = [], name = "n", check = True):
  
    if check:
        assert is_tidy_frame(df)
    
    if not isinstance(cols, (str, list)):
        raise Exception("cols should be a string or a list of strings")
      
    if isinstance(cols, str):
        cols = [cols]
        
    grouped_flag = is_grouped_frame(df, check = False)
    group_cols   = group_vars(df)
    if grouped_flag:
        df_ungrouped = df.obj
    else:
        df_ungrouped = df
         
    new_group_cols = list(set(group_cols + cols))
  
    if name in new_group_cols:
        raise Exception("count column name should be different from groupby variable names")
  
    # count by looking at grouped sizes
    # TODO: this should be rewritten with tidy verbs
    res = (df_ungrouped.groupby(new_group_cols)
                       .size()
                       .reset_index()
                       .rename(columns = {0: name}))
  
    if grouped_flag:
        res = res.groupby(group_cols)
  
    return res

#---------------------------------------------
def group_by(df, column_names, check = True):
    
    if check:
        assert is_tidy_frame(df)
    assert not is_grouped_frame(df)
    
    res = df.groupby(by = column_names
                     , sort = False
                     , observed = True
                     )
    
    return res

#---------------------------------------------
def get_colnames(df, check = True):
    if check:
        assert is_tidy_frame(df)
    if is_grouped_frame(df, check = False):
        res = list(df.obj.columns)
    else:
        res = list(df.columns)
    
    return res
    
#---------------------------------------------
def select(df, column_names, check = True):
    
    if check:
        assert is_tidy_frame(df)
    if isinstance(column_names, str):
        column_names = [column_names]
    # when column_names is a list, then check if everying is a string
    assert(all(map(lambda x: isinstance(x, str), column_names)))
    
    # pick grouping variables for a grouped frame
    if is_grouped_frame(df):
        groupvars    = group_vars(df)
        column_names = column_names + groupvars
        column_names = list(set(column_names))
        
        res = (df
               .pipe(ungroup)
               .loc[:, column_names]
               .pipe(group_by, groupvars, check = False)
               )
    else:
        res = df.loc[:, column_names]
        
    return(res)
    
#---------------------------------------------
def rename(df, names_dict, check = True):
    
    if check:
        assert is_tidy_frame(df)
    
    # pick grouping variables for a grouped frame
    if is_grouped_frame(df):
        groupvars = group_vars(df)
        group_common_vars = set(names_dict.keys()).intersection(groupvars)
        if len(group_common_vars) > 0:
          raise Exception("Cannot rename variables used in grouping")
        
        res = (df
               .pipe(ungroup)
               .loc[:, column_names]
               .pipe(group_by, groupvars, check = False)
               )
    else:
        res = df.rename(names_dict)
        
    return(res)

#---------------------------------------------
def summarize(df, name_func_dict, check = True):
    
    if check:
        assert is_tidy_frame(df)
    
    summarized_output = { key: None for key in name_func_dict.keys() }
    if is_grouped_frame(df):
        for key in name_func_dict:
            summarized_output[key] = df.apply(name_func_dict[key])
        res = pd.concat(summarized_output, axis = 1, join = "outer")
        res.columns = name_func_dict.keys()
        res = res.reset_index()
    else:
        for key in name_func_dict:
            summarized_output[key] = df.pipe(name_func_dict[key])
        res = pd.DataFrame(summarized_output, index = [0])
    
    return(res)

#---------------------------------------------
def summarize_across(df, cols, name_func_dict, check = True):
  
    if check:
          assert is_tidy_frame(df)
    
    summarized_output = { col: None for col in cols }
    
    if is_grouped_frame(df):
        print("todo") # TODO
    else:
        for col in cols:
            summarized_output[col] = list(name_func_dict.values())[0](df.loc[:, col])
        res = pd.DataFrame(summarized_output, index = [0])
    
    return(res)

#---------------------------------------------
def mutate(df, name_func_dict, check = True):
    
    if check:
        assert is_tidy_frame(df)
    
    if is_grouped_frame(df):
        groupvars = group_vars(df)
        
        def assigner(chunk, name):
            chunk[name] = name_func_dict[key](chunk)
            return chunk
        
        for key in name_func_dict:
            df = (df.apply(assigner, name = key)
                    .reset_index(drop = True)
                    .pipe(group_by, groupvars)
                    )
    else:
        for key in name_func_dict:
            df[key] = name_func_dict[key](df)
    
    return(df)

#---------------------------------------------
def arrange(df, cols, decreasing = False, check = True):
    
    if check:
        assert is_tidy_frame(df)
        
    if isinstance(cols, str):
      cols = [cols]
    
    if is_grouped_frame(df):
        groupvars = group_vars(df, check = False)
        res = (df.apply(lambda x: x.sort_values(by = cols
                                                , ascending = not decreasing
                                                )
                        )
                 .reset_index(drop = True)
                 .groupby(groupvars)
                 ) 
    else:
        res = df.sort_values(by = cols
                             , ascending = not decreasing
                             )
    
    return res

#---------------------------------------------
def filter(df, query_string = None, mask = None, check = True):
   
    if check:
        assert is_tidy_frame(df)
    
    if query_string is None and mask is None:
        raise Exception("Both 'query' and 'mask' cannot be None")
    if query_string is not None and mask is not None:
        raise Exception("One among 'query' and 'mask' should be None")
    if query_string is not None and mask is None:
        if is_grouped_frame(df):
            groupvars = group_vars(df, check = False)
            res = (df.obj
                     .query(query_string)
                     .groupby(groupvars)
                     )
        else:
            res = df.query(query_string)
    if query_string is None and mask is not None:
        if is_grouped_frame(df):
            raise Exception("'mask' does not work with grouped dataframe")
        res = df.iloc[mask, :]
        
    return res

#---------------------------------------------
def slice(df, row_numbers, check = True):
    
    if check:
        assert is_pandas_frame(df)
        assert is_tidy_frame(df)
    
    row_numbers = list(row_numbers)
    
    if is_grouped_frame(df):
        groupvars = group_vars(df, check = False)
        res = df.apply(lambda x: x.take(row_numbers))
        
        # try-catch block is to handle the case when the grouping
        # column(s) are already present and not insertable
        try:
            res = res.reset_index()
        except ValueError:
            res = res.reset_index(drop = True)
        
        res = res.groupby(groupvars)
    else:
        res = df.take(row_numbers)
    
    return(res)

#---------------------------------------------

is_pandas_frame(training)
is_tidy_frame(training)
is_grouped_frame(training)
count(training, "n_rides")

count_df = (training.pipe(count, ["n_rides", "is_dormant"])
                    .groupby(["n_rides"])
                    .pipe(select, ["n", "is_dormant"]))

count_df.pipe(ungroup)

temp = training.groupby("n_rides")
training.pipe(count, ["n_rides", "is_dormant"])
(training.pipe(group_by, "n_rides")
         .pipe(select, "ats", check = False)
         .head()
training.pipe(select, "n_rides")

summarize(training
          , {"mean_rides" : lambda x: np.mean(x["n_rides"]) + np.cos(x["ats"]),
             "median_ats" : lambda x: np.median(x["ats"])}
          )

(training.groupby(["is_dormant", "n_days"])
         .pipe(summarize
               , {"mean_rides" : lambda x:  np.mean(x["n_rides"]) + np.quantile(x["ats"], 0.7),
                  "median_ats" : lambda x: np.median(x["ats"])}
               )
         .pipe(ungroup)
         )

training.pipe(summarize_across, ["n_rides", "ltv"], {"mean": np.mean})

mutate(training
       , {"n_rides": lambda x: x["n_rides"] + 1}
       )

(training.pipe(group_by, ['is_dormant', 'peak_even_hours'])
         .pipe(mutate, {"n_rides": lambda x: x["n_rides"] + np.mean(x["n_rides"])})
         .pipe(ungroup)
         )

training.pipe(arrange, cols = ["n_rides", "is_dormant"])
training.groupby("is_dormant").pipe(arrange, by = ["n_rides"])
         
training.pipe(filter, "n_rides == 2")
training.pipe(filter, mask = (training["n_rides"] == 2).tolist())

(training.pipe(group_by, "is_dormant")
         .pipe(filter, "n_rides == 2")
         .head()
         )

training.pipe(slice, [0, 3])
training.groupby("n_rides").pipe(slice, [0]).head()




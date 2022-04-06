import copy as util_copy
import warnings
import re
import functools
import string as string
from collections import namedtuple

import numpy as np
import pandas as pd
from pandas.io.formats import format as fmt
from pandas._config import get_option
from collections_extended import setlist
from skimpy import skim
import pandas.api.types as dtypes

from tidypandas.tidy_utils import simplify, is_simple
from tidypandas._unexported_utils import (
                                            _is_kwargable,
                                            _is_valid_colname,
                                            _is_string_or_string_list,
                                            _enlist,
                                            _get_unique_names,
                                            _is_unique_list,
                                            _get_dtype_dict,
                                            _generate_new_string,
                                            _coerce_series,
                                            _coerce_pdf
                                        )
from tidypandas import tidyframe

@pd.api.extensions.register_dataframe_accessor("tp")
class tp:
    '''
    See the documentation of tidyframe class
    '''
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not is_simple(obj, verbose = True):
            raise Exception(("Input pandas dataframe is not 'simple'. Try "
                             "`tidypandas.tidy_utils.simplify` function."
                             ))
        
    @property
    def nrow(self):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.nrow  
    
    @property
    def ncol(self):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.ncol
    
    @property
    def shape(self):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.shape
      
    @property
    def dim(self):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.dim
        
    @property
    def colnames(self):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.colnames
    
    def add_row_number(self, name = 'row_number', by = None):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.add_row_number(name = name, by = by).to_pandas(copy = False)
    
    def add_group_number(self, by = None, name = 'group_number'):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.add_group_number(by = by, name = name).to_pandas(copy = False)
    
    def group_modify(self
                     , func
                     , by
                     , preserve_row_order = False
                     , row_order_column_name = "rowid_temp"
                     , is_pandas_udf = False
                     , **kwargs
                     ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.group_modify(func = func
                               , by = by
                               , preserve_row_order = preserve_row_order
                               , row_order_column_name = row_order_column_name
                               , is_pandas_udf = is_pandas_udf
                               , **kwargs
                               ).to_pandas(copy = False)
     
    def select(self, column_names = None, predicate = None, include = True):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.select(column_names = column_names
                         , predicate = predicate
                         , include = include
                         ).to_pandas(copy = False)
    
    def relocate(self, column_names, before = None, after = None):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.relocate(column_names = column_names
                           , before = before
                           , after = after
                           ).to_pandas(copy = False)
                           
    def rename(self, old_new_dict):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.rename(old_new_dict = old_new_dict).to_pandas(copy = False)
    
    
    def slice(self, row_numbers, by = None):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.slice(row_numbers = row_numbers
                        , by = by
                        ).to_pandas(copy = False)
    
    def arrange(self
                , order_by
                , na_position = 'last'
                , by = None
                ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.arrange(order_by = order_by
                          , na_position = na_position
                          , by = by
                          ).to_pandas(copy = False)

    def filter(self, query = None, mask = None, by = None, **kwargs):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.filter(query = query
                         , mask = mask
                         , by = by
                         , **kwargs
                         ).to_pandas(copy = False)
    
    def distinct(self, column_names = None, keep_all = False):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.distinct(column_names = column_names
                           , keep_all = keep_all
                           ).to_pandas(copy = False)
                           
    def mutate(self
               , dictionary = None
               , func = None
               , column_names = None
               , predicate = None
               , prefix = ""
               , by = None
               , order_by = None
               , **kwargs
               ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.mutate(dictionary = dictionary
                         , func = func
                         , column_names = column_names
                         , predicate = predicate
                         , prefix = prefix
                         , by = by
                         , order_by = order_by
                         , **kwargs
                         ).to_pandas(copy = False)
                           
    def summarise(self
                  , dictionary = None
                  , func = None
                  , column_names = None
                  , predicate = None
                  , prefix = ""
                  , by = None
                  , **kwargs
                  ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.summarise(dictionary = dictionary
                            , func = func
                            , column_names = column_names
                            , predicate = predicate
                            , prefix = prefix
                            , by = by
                            , **kwargs
                            ).to_pandas(copy = False)
                            
    summarize = summarise                        
    
    def inner_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.inner_join(y
                             , on = on
                             , on_x = on_x
                             , on_y = on_y
                             , sort = sort
                             , suffix_y = suffix_y
                             ).to_pandas(copy = False)
    
    def full_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.full_join(y
                             , on = on
                             , on_x = on_x
                             , on_y = on_y
                             , sort = sort
                             , suffix_y = suffix_y
                             ).to_pandas(copy = False)
    
    outer_join = full_join
    
    def left_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.left_join(y
                             , on = on
                             , on_x = on_x
                             , on_y = on_y
                             , sort = sort
                             , suffix_y = suffix_y
                             ).to_pandas(copy = False)
                             
    def right_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.right_join(y
                             , on = on
                             , on_x = on_x
                             , on_y = on_y
                             , sort = sort
                             , suffix_y = suffix_y
                             ).to_pandas(copy = False)
    
    def semi_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.semi_join(y
                             , on = on
                             , on_x = on_x
                             , on_y = on_y
                             , sort = sort
                             , suffix_y = suffix_y
                             ).to_pandas(copy = False)
                           
    def anti_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.anti_join(y
                             , on = on
                             , on_x = on_x
                             , on_y = on_y
                             , sort = sort
                             , suffix_y = suffix_y
                             ).to_pandas(copy = False)
    
    def cross_join(self, y, sort = True, suffix_y = "_y"):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.cross_join(y = y
                             , sort = sort
                             , suffix_y = suffix_y
                             ).to_pandas(copy = False)
    
    def cbind(self, y):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.cbind(y).to_pandas(copy = False)
    
    def rbind(self, y):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.rbind(y).to_pandas(copy = False)
    
    def count(self
              , column_names = None
              , name = 'n'
              , decreasing = True
              , wt = None
              ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.count(column_names = column_names
                        , name = name
                        , decreasing = decreasing
                        , wt = wt
                        ).to_pandas(copy = False)
    
    def add_count(self
                  , column_names = None
                  , name = 'n'
                  , wt = None
                  ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.add_count(column_names = column_names
                            , name = name
                            , wt = wt
                            ).to_pandas(copy = False)
    
    def pivot_wider(self
                    , names_from
                    , values_from
                    , values_fill = None
                    , values_fn = None
                    , id_cols = None
                    , sep = "__"
                    ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.pivot_wider(names_from = names_from
                              , values_from = values_from
                              , values_fill = values_fill
                              , values_fn = values_fn
                              , id_cols = id_cols
                              , sep = sep
                              ).to_pandas(copy = False)
    
    def pivot_longer(self
                     , cols
                     , names_to = "name"
                     , values_to = "value"
                     , include = True
                     , values_drop_na = False
                     ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.pivot_longer(cols = cols
                               , names_to = names_to
                               , values_to = values_to
                               , include = include
                               , values_drop_na = values_drop_na
                               ).to_pandas(copy = False)
    
    def slice_head(self
                   , n = None
                   , prop = None
                   , rounding_type = "round"
                   , by = None
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.slice_head(n = n
                             , prop = prop
                             , rounding_type = rounding_type
                             , by = by
                             ).to_pandas(copy = False)
    
    head = slice_head
    
    def slice_tail(self
                   , n = None
                   , prop = None
                   , rounding_type = "round"
                   , by = None
                   ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.slice_tail(n = n
                             , prop = prop
                             , rounding_type = rounding_type
                             , by = by
                             ).to_pandas(copy = False)                         
    
    tail = slice_tail
    
    def slice_sample(self
                     , n            = None
                     , prop         = None
                     , replace      = False
                     , weights      = None
                     , random_state = None
                     , by           = None
                     ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.slice_sample(n              = n
                               , prop         = prop
                               , replace      = replace
                               , weights      = weights
                               , random_state = random_state
                               , by           = by
                               ).to_pandas(copy = False)
    
    sample = slice_sample
    
    def slice_min(self
                  , n = None
                  , prop = None
                  , order_by_column = None
                  , with_ties = True
                  , rounding_type = "round"
                  , by = None
                  ):
        tf = tidyframe(self._obj, copy = False, check = False) 
        return tf.slice_min(n = n
                            , prop = prop
                            , order_by_column = order_by_column
                            , with_ties = with_ties
                            , rounding_type = rounding_type
                            , by = by
                            ).to_pandas(copy = False)
    
    def slice_max(self
                  , n = None
                  , prop = None
                  , order_by_column = None
                  , with_ties = True
                  , rounding_type = "round"
                  , by = None
                  ):
        tf = tidyframe(self._obj, copy = False, check = False) 
        return tf.slice_max(n = n
                            , prop = prop
                            , order_by_column = order_by_column
                            , with_ties = with_ties
                            , rounding_type = rounding_type
                            , by = by
                            ).to_pandas(copy = False)
                            
    def union(self, y):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.union(y).to_pandas(copy = False)
    
    def intersection(self, y):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.intersection(y).to_pandas(copy = False)
    
    def setdiff(self, y):
        tf = tidyframe(self._obj, copy = False, check = False)
        y  = tidyframe(y, copy = False, check = True)
        return tf.setdiff(y).to_pandas(copy = False)
    
    
    def any_na(self):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.any_na()
    
    def replace_na(self, value):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.replace_na(value = value).to_pandas(copy = False)
    
    def drop_na(self, column_names = None):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.drop_na(column_names = column_names).to_pandas(copy = False)
    
    def fill_na(self
                , column_direction_dict
                , order_by = None
                , ascending = True
                , na_position = "last"
                , by = None
                ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.fill_na(column_direction_dict = column_direction_dict
                          , order_by = order_by
                          , ascending = ascending
                          , na_position = na_position
                          , by = by
                          ).to_pandas(copy = False)
    
    fill = fill_na
    
    def separate(self
                 , column_name
                 , into
                 , sep = '[^[:alnum:]]+'
                 , strict = True
                 , keep = False
                 ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.separate(column_name = column_name
                           , into = into
                           , sep = sep
                           , strict = strict
                           , keep = keep
                           ).to_pandas(copy = False)
                           
    def unite(self, column_names, into, sep = "_", keep = False):
        tf = tidyframe(self._obj, copy = False, check = False) 
        return tf.unite(column_names = column_names
                        , into = into
                        , sep = sep
                        , keep = keep
                        ).to_pandas(copy = False)
    
    def separate_rows(self, column_name, sep = ';'):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.separate_rows(column_name = column_name
                                , sep = sep
                                ).to_pandas(copy = False)
    
    def nest_by(self
                , by = None
                , nest_column_name = 'data'
                , drop_by = True
                ):
        tf = tidyframe(self._obj, copy = False, check = False) 
        return tf.nest_by(by = by
                          , nest_column_name = nest_column_name
                          , drop_by = drop_by
                          ).to_pandas(copy = False)
                           
    def nest(self
             , column_names = None
             , nest_column_name = 'data'
             , include = True
             ):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.nest(column_names = column_names
                       , nest_column_name = nest_column_name
                       , include = include
                       ).to_pandas(copy = False)
    
    def unnest(self, nest_column_name = 'data'):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.unnest(nest_column_name = nest_column_name).to_pandas(copy = False)
    
    
    def split(self, by):
        tf = tidyframe(self._obj, copy = False, check = False)
        return tf.split(by = by).to_pandas(copy = False)
    
    group_split = split
    

# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------

import copy as util_copy
import warnings
import re
import functools
import importlib
import string as string
from collections import namedtuple

import numpy as np
import pandas as pd
from pandas.io.formats import format as fmt
from pandas._config import get_option
from collections_extended import setlist

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
import tidypandas.format as tidy_fmt


class tidyframe:
    '''
    tidyframe class
    A tidy pandas dataframe is a wrapper over 'simple' pandas 
    DataFrame object with method similar to tidyverse.
    
    Notes
    -----
    
    A pandas dataframe is said to be 'simple' if:
    
    1. Column names (x.columns) are an unnamed pd.Index object of unique 
       strings.
    2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0
       and step = 1.
    
    * Methods constitute a grammar of data manipulation mostly returning a 
      tidy dataframe as a result. 
    * When a method returns a tidy dataframe, it returns a copy and not 
      a view, unless copy is set to False.
    * Methods 'to_pandas' and 'to_series' convert into pandas dataframe or 
      series.
    * The only attribute of the class is the underlying pandas dataframe. This 
      cannot be accessed by the user. Please use 'to_pandas' method to obtain 
      a copy of the underlying pandas dataframe.
    
    Attributes
    ----------
    nrow: Number of rows
    ncol: Number of columns
    shape(alias dim): tuple of number of rows and number of columns
    colnames: list of column names
    
    Methods
    -------
    init
        Creates a tidy dataframe from a 'simple' pandas dataframe
        
    'to' methods:
        to_pandas
            Returns the underlying pandas dataframe
        to_series
            Returns selected column as pandas series
    
    'pipe' methods:
        pipe    
            A pipe method for tidy dataframe
        pipe_tee
            A pipe method called for side-effect and returns the input

    'basic' methods:
        select
            Returns a dataframe with selected columns
        arrange
            Returns a dataframe after sorting rows
        slice
            Returns a dataframe to subset by row numbers
        distinct
            Returns a distinct rows defined by selected columns
        filter
            Returns a dataframe with rows selected by some criteria
        mutate
            Returns a dataframe by adding or changing a few columns
        summarise (alias: summarize)
            Returns a dataframe after aggregating over selected columns
        
    'count' methods:
        count
            Returns a dataframe after counting over selected columns
        add_count
            Returns a dataframe by adding a new count column to input
            
    'pivot' methods:
        pivot_wider
            Returns a dataframe by converting from long to wide format
        pivot_longer
            Returns a dataframe by converting from wide to long format
            
    'missing value' methods:
        drop_na
            Returns a dataframe by dropping rows which have mssing values in
            selected columns
        replace_na
            Returns a dataframe by replacing missing values in selected columns
        fill_na
            Returns a dataframe by filling missing values from up, down or 
            both directions for selected columns
    'string' methods:
        separate
            Returns a dataframe by splitting a string column into multiple 
            columns
        unite
            Returns a dataframe by combining multiple string columns 
        separate_rows
            Returns a dataframe by exploding a string column
    'slice' extensions:
        slice_head
            Returns a dataframe with top few rows of the input
        slice_tail
            Returns a dataframe with last few rows of the input
        slice_max
            Returns a dataframe with few rows corrsponding to maximum values of
            some columns
        slice_min
            Returns a dataframe with few rows corrsponding to maximum values of
            some columns
        slice_sample
            Returns a dataframe with a sample of rows
    
    'join' methods:
        join, inner_join, outer_join, left_join, right_join, anti_join:
            Returns a joined dataframe of a pair of dataframes
    'set operations' methods:
        union, intersection, setdiff:
            Returns a dataframe after set like operations over a pair of 
            dataframes
    'bind' methods:
        rbind, cbind:
            Returns a dataframe by rowwise or column wise binding of a pair of 
            dataframes
    
    'misc' methods:
        add_row_number:
            Returns a dataframe with rowids added
        add_group_number:
            Returns a dataframe with group ids added
    
    '''
    
    ##########################################################################
    # init
    ##########################################################################
    
    def __init__(self, x, check = True, copy = True):
        '''
        init
        Create tidy dataframe from a 'simple' ungrouped pandas dataframe

        Parameters
        ----------
        x : 'simple' pandas dataframe
        check : bool, Default is True
            Whether to check if the input pandas dataframe is 'simple'
        copy: bool, Default is True
            Whether the tidyframe object to be created should refer 
            to a copy of the input pandas dataframe or the input itself

        Notes
        -----
        1. A pandas dataframe is said to be 'simple' if:
            a. Column names (x.columns) are an unnamed pd.Index object of 
               unique strings.
            b. Row names (x.index) are an unnamed pd.RangeIndex object with 
               start = 0 and step = 1.
        2. Unless you are sure, in general usage:
            a. set arg 'check' to `True`.
            b. set arg `copy` to `True`.
        
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> penguins_tidy
        '''
        assert isinstance(check, bool),\
            "arg 'check' should be a bool"
        assert isinstance(copy, bool),\
            "arg 'copy' should be a bool"
        if check:
            if not is_simple(x, verbose = True):
                try:
                    x = simplify(x, verbose = True)
                except:
                    raise Exception(("Input pandas dataframe could not be simplified"
                                     " See to above warnings."
                                    ))
        
        if copy:                       
            self.__data = (x.copy()
                            .convert_dtypes()
                            .fillna(pd.NA)
                            .pipe(_coerce_pdf)
                            )
        else:
            self.__data = x
        return None
    
    ##########################################################################
    # repr
    ##########################################################################
    # def __repr__(self):
        
    #     nr = self.nrow
    #     nc = self.ncol
        
    #     header_line   = f"# A tidy dataframe: {nr} X {nc}"
    #     head_10 = self.__data.head(10)
    #     # dtype_dict = _get_dtype_dict(self.__data)
    #     # for akey in dtype_dict:
    #     #     dtype_dict[akey] = akey +  " (" + dtype_dict[akey][0:3] + ")"
    #     # 
    #     # head_10 = self.__data.head(10).rename(columns = dtype_dict)
    #     pandas_str = head_10.__str__()

    #     left_over = nr - 10
    #     if left_over > 0:
    #         leftover_str = f"# ... with {left_over} more rows"

    #         tidy_string = (header_line +
    #                        '\n' +
    #                        pandas_str +
    #                        '\n' +
    #                        leftover_str
    #                        )
    #     else:
    #         tidy_string = (header_line +
    #                        '\n' +
    #                        pandas_str
    #                        )

    #     return tidy_string

    def __repr__(self) -> str:
        """
        Return a string representation for a particular DataFrame.
        """
        # if self._info_repr():
        #     buf = StringIO()
        #     self.info(buf=buf)
        #     return buf.getvalue()

        repr_params = fmt.get_dataframe_repr_params()
        # return self.to_string(**repr_params)

        nr = self.nrow
        nc = self.ncol

        header_line   = f"# A tidy dataframe: {nr} X {nc}"

        from pandas import option_context

        with option_context("display.max_colwidth",repr_params["max_colwidth"]):
            row_truncated = False
            show_frame = self.__data
            if repr_params["min_rows"] and self.__data.shape[0] > repr_params["min_rows"]:
                row_truncated = True
                show_frame = self.__data.iloc[:repr_params["min_rows"], :]

            formatter = tidy_fmt.TidyDataFrameFormatter(
                # self.__data,
                show_frame,
                # min_rows=repr_params["min_rows"],
                max_rows=repr_params["max_rows"],
                max_cols=repr_params["max_cols"],
                # show_dimensions=repr_params["show_dimensions"]
            )
            formatted_str = header_line + "\n" + fmt.DataFrameRenderer(formatter)\
                                               .to_string(line_width=repr_params["line_width"])
            if row_truncated:
                footer_str = "#... with {} more rows".format(self.__data.shape[0]-repr_params["min_rows"])
                formatted_str += "\n" + footer_str
            return formatted_str

    # def _repr_html_(self):
    #     return self.__data._repr_html_()

    def _repr_html_(self):
        """
        Return a html representation for a particular DataFrame.
        Mainly for IPython notebook.
        """
        # if self._info_repr():
        #     buf = StringIO()
        #     self.info(buf=buf)
        #     # need to escape the <class>, should be the first line.
        #     val = buf.getvalue().replace("<", r"&lt;", 1)
        #     val = val.replace(">", r"&gt;", 1)
        #     return "<pre>" + val + "</pre>"
        
        if get_option("display.notebook_repr_html"):
            max_rows = get_option("display.max_rows")
            min_rows = get_option("display.min_rows")
            max_cols = get_option("display.max_columns")
            show_dimensions = get_option("display.show_dimensions")

            formatter = tidy_fmt.TidyDataFrameFormatter(
                self.__data,
                columns=None,
                col_space=None,
                na_rep="<NA>",
                formatters=None,
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                escape=True,
                max_rows=max_rows,
                min_rows=min_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=".",
            )
            return tidy_fmt.TidyDataFrameRenderer(formatter).to_html(notebook=True)
        else:
            return None


    
    ##########################################################################
    # to_pandas methods
    ##########################################################################
    
    def to_pandas(self, copy = True):
        '''
        Return copy of underlying pandas dataframe
        
        Parameters
        ----------
        copy: bool, default is True
            Whether to return a copy of pandas dataframe held by
            tidyframe object or to return the underlying
            pandas dataframe itself
        
        Returns
        -------
        pandas dataframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        
        >>> penguins_tidy.to_pandas()
        >>> # check whether the dataframes are same
        >>> penguins.equals(penguins_tidy.to_pandas())
        '''
        assert isinstance(copy, bool),\
            "arg 'copy' should be a bool"
        if copy:
            res = self.__data.copy()
        else:
            res = self.__data
            
        return res
    
    ##########################################################################
    # to_series or pull
    ##########################################################################
    def pull(self, column_name = None, copy = True):
        '''
        pull (aka to_series)
        Returns a copy of column as pandas series
        
        Parameters
        ----------
        column_name : str or None
            Name of the column to be returned as pandas series. When there is
            only one column, it can be None.
        copy: bool, default is True
            Whether to return a copy of the pandas series object

        Returns
        -------
        pandas series
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> penguins_tidy.pull("species")
        '''
        
        if column_name is not None:
            assert isinstance(column_name, str),\
                "arg 'column_name'should be a string"
        if self.ncol > 1:
            assert isinstance(column_name, str),\
                "Column to pull should be specified"
        else:
            if column_name is None:
                column_name = self.colnames[0]
        
        assert column_name in list(self.__data.columns), \
            "arg 'column_name' should be an existing column name"
        assert isinstance(copy, bool),\
            "arg 'copy' should be a bool"
        
        if copy:
            res = self.__data[column_name].copy()
        else:
            res = self.__data[column_name]
        return res
    
    to_series = pull
    
    ##########################################################################
    # pipe methods
    ##########################################################################
    
    # pipe
    def pipe(self, func, *args, **kwargs):
        '''
        Returns func(self)

        Parameters
        ----------
        func : callable

        Returns
        -------
        Depends on the return type of `func`
        '''
        assert callable(func),\
            "arg 'func' should be callable"
        
        return func(self, *args, **kwargs)
    
    # pipe_tee
    def pipe_tee(self, func, *args, **kwargs):
        '''
        pipe for side-effect

        Parameters
        ----------
        func : callable

        Returns
        -------
        tidyframe
        '''
        assert callable(func),\
            "arg 'func' should be callable"
            
        func(self, *args, **kwargs) # side-effect
        
        return self
    
    
    ##########################################################################
    # attributes
    ##########################################################################
            
    @property
    def nrow(self):
        return self.__data.shape[0]
    
    @property
    def ncol(self):
        return self.__data.shape[1]
    
    @property
    def shape(self):
        return self.__data.shape
      
    @property
    def dim(self):
        return self.__data.shape
        
    @property
    def colnames(self):
        return list(self.__data.columns)
    
    ##########################################################################
    # summarizers -- skim, glimpse
    ##########################################################################
    
    def skim(self, return_self = False):
        '''
        Skim the tidy dataframe.
        Provides a meaningful summary of the dataframe

        Parameters
        ----------
        return_self : bool (default is False)
            Whether to return the input
            This is helpful while piping the input to next method while skim is
            a side-effect
            
        Returns
        -------
        None when return_self is False, input otherwise
        
        Notes
        -----
        1. skim is done via 'skimpy' package is expected to be installed.
        '''
        # TODO: Implement package check
        if importlib.util.find_spec("skimpy") is not None:
            from skimpy import skim
            skim(self.__data)

        if return_self:
            return self
        else:
            return None
        
    ##########################################################################
    # validators
    ##########################################################################
    
    def _validate_by(self, by):
        
        assert _is_string_or_string_list(by),\
            "arg 'by' should be a string or a list of strings"
            
        by = _enlist(by)
        
        assert len(set(by)) == len(by),\
            "arg 'by' should have unique strings"
        
        assert set(self.colnames).issuperset(by),\
            "arg 'by' should contain valid column names"
            
        return None
    
    def _validate_column_names(self, column_names):
        
        assert _is_string_or_string_list(column_names),\
            "arg 'column_names' should be a string or a list of strings"
            
        column_names = _enlist(column_names)
        
        assert len(set(column_names)) == len(column_names),\
            "arg 'column_names' should have unique strings"
        
        cols_not_found = list(set(column_names).difference(self.colnames))
        assert set(self.colnames).issuperset(column_names),\
            (f"arg 'column_names' should contain valid column names"
             f"These column(s) do not exist: {cols_not_found}"
            )
        
        return None
      
    def _clean_order_by(self, order_by):
        
        cns = self.colnames
        if isinstance(order_by, str):
            order_by = _enlist(order_by)
            
        assert isinstance(order_by, (list, tuple)),\
            "'order_by' should be a string, list or a tuple"
        
        if isinstance(order_by, tuple):
            order_by = list(order_by)
        
        for id, x in enumerate(order_by):           
            if isinstance(x, tuple):
                order_by[id] = list(x)
            elif isinstance(x, str):
                order_by[id] = [x, 'asc']
            elif isinstance(x, list):
                pass
            else:
                raise Exception(("An element of 'order_by' should be a list "
                                 "or a tuple or a string"))
        
        for id, x in enumerate(order_by):
            assert (len(x) <= 2) and (len(x) >= 1),\
                "An element of 'order_by' should have length between 1 and 2"
            assert x[0] in cns,\
                f"Non-existing column name provided in 'order_by'. Input: {x[0]}"
            if len(x) == 1:
                x.append('asc')
                order_by[id] = 'asc'
            assert x[1] in ['asc', 'desc'],\
                (f"Ordering specification in 'order_by' should be one among "
                 f"('asc', 'desc'). Input: {x[1]}"
                 )
            
        # check for unique names
        assert _is_unique_list([x[0] for x in order_by]),\
            "Ordering columns should be unique"
        
        return order_by
    
    ##########################################################################
    # add_row_number
    ##########################################################################
    def add_row_number(self, name = 'row_number', by = None):
        '''
        add_row_number (aka rowid_to_column)
        Add a row number column to tidyframe
        
        Parameters
        ----------
        name : str
            Name for the row number column
        by : str or list of strings
            Columns to group by
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Row order is preserved.
        2. Column indicating row number is added as the first column 
           (to the left).
        3. Alias: rowid_to_column
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())

        >>> penguins_tidy.add_row_number() # equivalently penguins_tidy.add_rowid()
        
        >>> # add row number per group in the order of appearance
        >>> penguins_tidy.add_row_number(by = 'sex')
        '''
        
        nr = self.nrow
        cn = self.colnames
        
        assert isinstance(name, str),\
            "arg 'name' should be a string"
        if name[0] == "_":
            raise Exception("'name' should not start with an underscore")
        
        if name in cn:
            raise Expection("'name' should not be an existing column name.")
            
        if by is None:
            res = self.__data.assign(**{name : np.arange(nr)})
        else:
            self._validate_by(by)
            by = _enlist(by)
            res = (self
                   .__data
                   .assign(**{"_rn" : np.arange(nr)})
                   .groupby(by, sort = False, dropna = False)
                   .apply(lambda x: x.assign(**{name : np.arange(x.shape[0])}))
                   .reset_index(drop = True)
                   .sort_values("_rn", ignore_index = True)
                   .drop(columns = "_rn")
                   )
        
        col_order = [name] + cn
        
        return tidyframe(res.loc[:, col_order]
                                   , check = False
                                   , copy = False
                                   )
    
    rowid_to_column = add_row_number
    
    ##########################################################################
    # add_group_number
    ##########################################################################
    def add_group_number(self, by = None, name = 'group_number'):
        '''
        Add a group number column to tidyframe
        
        Parameters
        ----------
        name : str
            Name for the group number column
        by : str or list of strings
            Columns to group by
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Row order is preserved.
        2. Column indicating group number is added as the first column 
           (to the left).
        3. Number for the group is based on the first appearance of a
           group combination.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())

        >>> penguins_tidy.add_group_number(by = 'sex')
        '''
        
        nr = self.nrow
        cn = self.colnames
        
        assert isinstance(name, str),\
            "arg 'name' should be a string"
        if name[0] == "_":
            raise Exception("'name' should not start with an underscore")
        
        if name in cn:
            raise Exception("'name' should not be an existing column name.")
          
        self._validate_by(by)
        by = _enlist(by)
            
        group_order_frame = (
               self
               .__data
               .loc[:, by]
               .drop_duplicates()
               .assign(**{name: lambda x: np.arange(x.shape[0])})
               )
        
        row_name = _generate_new_string(cn + [name])
        
        res = (self.__data
                   .assign(**{row_name: lambda x: np.arange(x.shape[0])})
                   .merge(group_order_frame
                          , on = by
                          , how = 'left'
                          )
                   .sort_values(by = row_name)
                   .loc[:, [name] + cn]
                   .reset_index(drop = True)
                   )
        
        return tidyframe(res, check = False, copy = False)

    ##########################################################################
    # group_modify
    ##########################################################################
    def group_modify(self
                     , func
                     , by
                     , preserve_row_order = False
                     , row_order_column_name = "rowid_temp"
                     , is_pandas_udf = False
                     , **kwargs
                     ):
        '''
        Split by some columns, apply a function per chunk which returns a 
        dataframe and combine it back into a single dataframe
        
        Parameters
        ----------
        func: callable
            Type 1. A function: TidyDataFrame --> TidyDataFrame
            Type 2. A function: simple pandas df --> simple pandas df
            In latter case, set 'is_pandas_udf' to True
        
        by: string or list of strings
            Column names to group by
        
        preserve_row_order: bool (default is False)
            Whether to preserve the row order of the input dataframe
        
        row_order_column_name: string
            Temporary column name to be created to maintain row order
            
        is_pandas_udf: bool (default is False)
            Whether the 'func' argument is of type 2
            
        **kwargs: arguments to 'func'
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Chunks will always include the grouping columns.
        2. If grouping columns are found in output of 'func', then they are
           removed and replaced with value in input chunk.
        3. When 'preserve_row_order' is True, a temporary column is added
           to each chunk. udf should pass it through without modification.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # pick a sample of rows per chunk defined by 'species'
        >>> penguins_tidy.group_modify(lambda x: x.sample(n = 3)
        >>>                             , by = 'species'
        >>>                             )
        >>> 
        >>> # apply a pandas udf per chunk defined by 'species'
        >>> # groupby columns are always added to the left
        >>> penguins_tidy.group_modify(lambda x: x.select('year')
        >>>                             , by = ['species', 'sex']
        >>>                             )
        >>>                                 
        >>> # preserve row order
        >>> # a temporary column (default: 'rowid_temp') is added to each chunk
        >>> # udf should not meddle with the temporary column
        >>> (penguins_tidy
        >>>     .select('sex')
        >>>     # add 'row_number' column to illustrate row preservation
        >>>     .add_row_number()
        >>>     # print the output of each chunk
        >>>     # sample 2 rows
        >>>     .group_modify(lambda x: x.pipe_tee(print).sample(2)
        >>>                   , by = 'sex'
        >>>                   , preserve_row_order = True
        >>>                   )
        >>>   )
        >>>   
        >>> # use kwargs
        >>> penguins_tidy.group_modify(lambda x, **kwargs: x.sample(n = kwargs['size'])
        >>>                             , by = 'species'
        >>>                             , size = 3
        >>>                             )
        '''
        self._validate_by(by)
        by = _enlist(by)
        cn = self.colnames
        nr = self.nrow
        
        assert callable(func),\
            "'func' should be a function: tidy df --> tidy df"
        assert isinstance(preserve_row_order, bool),\
            "arg 'preserve_row_order' should be a bool"
        assert isinstance(row_order_column_name, str),\
            "arg 'row_order_column_name' should be a string"
        assert row_order_column_name not in self.colnames,\
            "arg 'row_order_column_name' should not be a existing column name"
        
        group_cn = _generate_new_string(cn)
        
        group_id_frame = self.__data.loc[:, by].drop_duplicates()
        group_id_frame[group_cn] = np.arange(group_id_frame.shape[0])
      
        # create wrapper function
        if is_pandas_udf:
            def wrapper_func(chunk, **kwargs):
                res = func(chunk.drop(columns = group_cn).reset_index(drop = True)
                           , **kwargs
                           )
                by_left = list(set(by).intersection(list(res.columns)))
                if len(by_left) > 0:
                    res.drop(columns = by_left, inplace = True)
                for col in by:
                    res[col] = chunk[col].iloc[0]
                
                cols_in_order = util_copy.copy(by)
                cols_in_order.extend(set(res.columns).difference(by))
                return res.loc[:, cols_in_order]
        else:
            def wrapper_func(chunk, **kwargs):
                # i/o are pdfs
                chunk_tidy = (tidyframe(chunk.reset_index(drop = True)
                                                 , check = False
                                                 , copy = False
                                                 )
                              .select(group_cn, include = False)
                              )
                res        = func(chunk_tidy, **kwargs).to_pandas(copy = False)
                by_left    = list(set(by).intersection(list(res.columns)))
                if len(by_left) > 0:
                    res.drop(columns = by_left, inplace = True)
                for col in by:
                    res[col] = chunk[col].iloc[0]
                
                cols_in_order = util_copy.copy(by)
                cols_in_order.extend(set(res.columns).difference(by))
                return res.loc[:, cols_in_order] 
        
        if preserve_row_order:
            
            assert row_order_column_name not in cn,\
                ("arg 'row_order_column_name' should not be an exisiting column "
                 "name"
                 )
            
            res = (self.__data
                        .assign(**{row_order_column_name: np.arange(nr)})
                        .merge(group_id_frame, on = by)
                        .groupby(group_cn, sort = False, dropna = False)
                        .apply(wrapper_func, **kwargs)
                        )
            
            try:
                res = (res.pipe(simplify)
                          .drop(columns = group_cn)
                          )
            except:
                raise Exception(("Result of apply is too complex to be simplified "
                                 "by `simplify` function"
                                 )
                                )
            
            if row_order_column_name in list(res.columns):
                res = (res.sort_values(row_order_column_name
                                       , ignore_index = True
                                       )
                          .drop(columns = row_order_column_name)
                          )
            else:
                raise Exception(
                    ("'row_order_column_name' in each chunk should "
                     "be retained, when 'preserve_row_order' is True"
                     ))
        else: # do not preserve row order
            res = (self.__data
                       .merge(group_id_frame, on = by)
                       .groupby(group_cn, sort = False, dropna = False)
                       .apply(wrapper_func, **kwargs)
                       )
            try:
                res = (res.pipe(simplify)
                          .drop(columns = group_cn)
                          )
            except:
                raise Exception(("Result of apply is too complex to be simplified "
                                 "by `simplify` function"))
        
        return tidyframe(res
                         , check = False
                         , copy = False
                         )
    
    ##########################################################################
    # basic verbs
    ##########################################################################
    
    
    ##########################################################################
    # select
    ##########################################################################
    
    def select(self, column_names = None, predicate = None, include = True):
        '''
        Select a subset of columns by name or predicate

        Parameters
        ----------
        column_names : str or a list of strings
            list of column names to be selected
        predicate : callable, series --> bool
            function which returns a bool
        include : bool, default is True
            When True, column_names are selected, else dropped

        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Select works by either specifying column names or a predicate,
           not both.
        2. When predicate is used, predicate should accept a pandas series and
           return a bool. Each column is passed to the predicate and the result
           indicates whether the column should be selected or not.
        
        Examples
        --------
        >>> import re
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # select with names
        >>> penguins_tidy.select(['sex', 'species'])
        >>> 
        >>> # select using a predicate: only non-numeric columns
        >>> penguins_tidy.select(predicate = lambda x: x.dtype != "string")
        >>> 
        >>> # select columns ending with 'mm'
        >>> penguins_tidy.select(
        >>>     predicate = lambda x: bool(re.match(".*mm$", x.name))
        >>>     )
        >>> 
        >>> # invert the selection
        >>> penguins_tidy.select(['sex', 'species'], include = False)
        '''
        
        cn = self.colnames
        
        if (column_names is None) and (predicate is None):
            raise Exception(("Exactly one among 'column_names' and 'predicate' "
                             "should not be None")
                            )
        if (column_names is not None) and (predicate is not None):
            raise Exception(("Exactly one among 'column_names' and 'predicate' "
                             "should not be None"))
        
        if column_names is None:
            assert callable(predicate),\
                "arg 'predicate'(series --> bool) should be a function"
            col_bool_list = np.array(list(self.__data.apply(predicate
                                                            , axis = "index"
                                                            )
                                         )
                                    )
            column_names = list(np.array(cn)[col_bool_list])
        else:
            self._validate_column_names(column_names)
            column_names = _enlist(column_names)
        
        if not include:
            column_names = list(setlist(cn).difference(column_names))
        
        if len(column_names) == 0:
            raise Exception("At least one column should be selected")
            
        res = self.__data.loc[:, column_names]
        # return a copy due to loc
        return tidyframe(res, check = False, copy = True)
    
    ##########################################################################
    # relocate
    ##########################################################################
    
    def relocate(self, column_names, before = None, after = None):
        '''
        relocate the columns of the tidy pandas dataframe

        Parameters
        ----------
        column_names : string or a list of strings
            column names to be moved
        before : string, optional
            column before which the column are to be moved. The default is None.
        after : TYPE, optional
            column after which the column are to be moved. The default is None.
        
        
        Returns
        -------
        tidyframe

        Notes
        -----
        Only one among 'before' and 'after' can be not None. When both are None,
        the columns are added to the begining of the dataframe (leftmost)
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # move "island" and "species" columns to the left of the dataframe
        >>> penguins_tidy.relocate(["island", "species"])
        >>> 
        >>> # move "sex" and "year" columns to the left of "island" column
        >>> penguins_tidy.relocate(["sex", "year"], before = "island")
        >>> 
        >>> # move "island" and "species" columns to the right of "year" column
        >>> penguins_tidy.relocate(["island", "species"], after = "year")
        '''
        
        self._validate_column_names(column_names) 
        column_names = _enlist(column_names)
        cn = self.colnames
         
        assert not ((before is not None) and (after is not None)),\
            "Atleast one arg among 'before' and 'after' should be None"
            
        if after is not None:
            assert isinstance(after, str),\
                "arg 'after' should be a string"
            assert after in cn,\
                "arg 'after' should be a exisiting column name"
            assert not (after in column_names),\
                "arg 'after' should be an element of 'column_names'"
        
        if before is not None:
            assert isinstance(before, str),\
                "arg 'before' should be a string"
            assert before in cn,\
                "arg 'before' should be a exisiting column name"
            assert not (before in column_names),\
                "arg 'before' should be an element of 'column_names'"
        
        cc_setlist       = setlist(cn)
        cc_trunc_setlist = cc_setlist.difference(column_names)
            
        # case 1: relocate to start when both before and after are None
        if (before is None) and (after is None):
            new_colnames = column_names + list(cc_trunc_setlist)
        elif (before is not None):
            # case 2: before is not None
            pos_before   = np.where(np.array(cc_trunc_setlist) == before)
            pos_before   = int(pos_before[0])
            cc_left      = list(cc_trunc_setlist[:pos_before])
            cc_right     = list(cc_trunc_setlist[pos_before:])
            new_colnames = cc_left + column_names + cc_right
        else:
            # case 3: after is not None
            pos_after   = np.where(np.array(cc_trunc_setlist) == after)
            pos_after   = int(pos_after[0])    
            cc_left      = list(cc_trunc_setlist[ :(pos_after + 1)])
            cc_right     = list(cc_trunc_setlist[(pos_after + 1): ])
            new_colnames = cc_left + column_names + cc_right
      
        res = self.__data.loc[:, new_colnames]
        # return a copy due to loc
        return tidyframe(res, check = False, copy = True)
    
    ##########################################################################
    # rename
    ##########################################################################
    
    def rename(self, old_new_dict):
        '''
        Rename columns of the tidy pandas dataframe
        
        Parameters
        ----------
        old_new_dict: A dict with old names as keys and new names as values
        
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())

        >>> penguins_tidy.rename({'species': 'species_2'})
        '''
        cn = self.colnames
        
        assert isinstance(old_new_dict, dict),\
            "arg 'old_new_dict' should be a dict"
        assert set(cn).issuperset(old_new_dict.keys()),\
            "keys of the input dict should be existing column names"
        assert _is_string_or_string_list(list(old_new_dict.values())),\
            "values of the dict should be strings"
        assert _is_unique_list(list(old_new_dict.values())),\
            "values of the dict should be unique"
        
        # new names should not intersect with 'remaining' names
        remaining = set(cn).difference(old_new_dict.keys())
        assert len(remaining.intersection(old_new_dict.values())) == 0,\
            ("New intended column names (values of the dict) lead to duplicate "
             "column names"
             )
        
        res = self.__data.rename(columns = old_new_dict)
        return tidyframe(res, check = False, copy = False)
    
    ##########################################################################
    # slice
    ##########################################################################
    
    def slice(self, row_numbers, by = None):
        '''
        Subset rows of a tidyframe
        
        Parameters
        ----------
        row_numbers : int or list or range or 1-D numpy array of positive integers
            list/array of row numbers.
        by : list of strings, optional
            Column names to groupby. The default is None.

        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Grouped slice preserves the row order.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())

        >>> # pick first three rows of the dataframe
        >>> penguins_tidy.slice(np.arange(3))
        >>> 
        >>> # pick these row numbers: [0, 3, 8]
        >>> penguins_tidy.slice([0, 3, 8])
        >>> 
        >>> # pick first three rows per specie
        >>> penguins_tidy.slice([0,1,2], by = "species")
        >>> 
        >>> # pick first three rows for each species and sex combination
        >>> penguins_tidy.slice(np.arange(3), by = ["species", "sex"])
        '''
        
        if isinstance(row_numbers, int):
            row_numbers = _enlist(row_numbers)
        
        if not isinstance(row_numbers, list):
            row_numbers = list(row_numbers)
            
        assert all([dtypes.is_integer(x) for x in row_numbers]),\
            "arg 'row_numbers' should be a list or array of positive integers"
        
        if by is None:
            minval = np.min(row_numbers)
            maxval = np.max(row_numbers)
            assert minval >= 0 and maxval <= self.nrow,\
                ("row numbers to slice should be in a range the max and min "
                 "rows of the dataframe"
                 )
            
            res = (self.__data
                       .iloc[row_numbers, :]
                       .reset_index(drop = True)
                       .pipe(lambda x: tidyframe(x, copy = True, check = False))
                       )
        else:
            self._validate_by(by)
            by = _enlist(by)
            
            min_group_size = (self.__data
                                  .groupby(by, sort = False, dropna = False)
                                  .size()
                                  .min()
                                  )
                                  
            if np.max(row_numbers) > min_group_size:
                print("Minimum group size is: ", min_group_size)
                raise Exception(("Maximum row number to slice per group should" 
                                 "not exceed the number of rows of the group"
                                 ))
            
            res = (self.group_modify(lambda chunk: chunk.iloc[row_numbers, :]
                                     , by = by
                                     , is_pandas_udf = True
                                     , preserve_row_order = True
                                     , row_order_column_name = _generate_new_string(self.colnames)
                                     )
                       .relocate(self.colnames)
                       )
        
        return res
    
    ##########################################################################
    # arrange
    ##########################################################################
        
    def arrange(self
                , order_by
                , na_position = 'last'
                , by = None
                ):
        '''
        Orders the rows by the values of selected columns
        
        Parameters
        ----------
        column_names : list of strings
            column names to order by.
        order_by: str, list, tuple
            column names and asc/desc tuples. See examples.
        na_position : str, optional
            One among: 'first', 'last'. The default is 'last'.
        by: str, list of strings (default: None)
            Column names to group by

        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # arrange by ascending order of column 'bill_length_mm'
        >>> penguins_tidy.arrange('bill_length_mm')
        >>> # equivalent to below:
        >>> # penguins_tidy.arrange([('bill_length_mm', 'asc')])
        >>> 
        >>> # arrange by descding order of column 'bill_length_mm'
        >>> penguins_tidy.arrange([('bill_length_mm', 'desc')])
        >>> 
        >>> # arrange by ascending order of column 'body_mass_g' and break ties with
        >>> # ascending order to 'bill_length_mm'
        >>> penguins_tidy.arrange([('body_mass_g', 'asc'), ('bill_length_mm', 'asc')])
        >>> # equivalent to below:
        >>> # penguins_tidy.arrange(['body_mass_g', 'bill_length_mm'])
        >>> 
        >>> # arrange by ascending order of column 'body_mass_g' and break ties with
        >>> # descending order to 'bill_length_mm'
        >>> penguins_tidy.arrange([('body_mass_g', 'asc'), ('bill_length_mm', 'desc')])
        >>> # equivalent to below:
        >>> penguins_tidy.arrange(['body_mass_g', ('bill_length_mm', 'desc')])
        >>> 
        >>> # determine where NA has to appear
        >>> penguins_tidy.arrange('sex', na_position = 'first')
        >>> 
        >>> # grouped arrange: rearranges within the group
        >>> penguins_tidy
        >>> penguins_tidy.arrange('bill_length_mm', by = 'sex')
        >>> # notice that order of 'sex' column does not change while row corresponding
        >>> # to 'bill_length_mm' per group is sorted in ascending order
        >>> # This preserves the relative position of groups
        >>> # If you intend to not preserve the row order, then simply
        >>> # include the grouping variable(s) in order
        >>> # ex: penguins_tidy.arrange(['sex', 'bill_length_mm'])
        
        Notes
        -----
        1. Grouped arrange rearranges the rows within a group without
        changing the relative position of groups in the dataframe.
        
        2. When multiple columns are provided in order_by, second column is
        used to break the ties after sorting by first column and so on.
        
        3. If the column(s) provided in arrange are not sufficient to order 
        the rows of the dataframe, then row number is implicitly used to
        deterministically break ties.
        '''
        
        cn = self.colnames
        nr = self.nrow
        
        order_by = self._clean_order_by(order_by)
        column_names = [x[0] for x in order_by]
        ascending = [True if x[1] == "asc" else False for x in order_by]
        
        if by is not None:
            self._validate_by(by)
            by = _enlist(by)
            assert len(set(by).intersection(column_names)) == 0,\
                "'by' and column names in 'order_by' should not have common names"
        
        def pdf_sortit(x):
            res = x.sort_values(by             = column_names
                                , axis         = 0
                                , ascending    = ascending
                                , inplace      = False
                                , kind         = 'quicksort'
                                , na_position  = na_position
                                , ignore_index = True
                                )
            return res
        
        if by is None:
            res = pdf_sortit(self.__data)
        else:
            rn_org = _generate_new_string(cn)
            g_id   = _generate_new_string(cn + [rn_org])
            wgn    = _generate_new_string(cn + [rn_org, g_id])
            rn_arr = _generate_new_string(cn + [rn_org, g_id, wgn])
            
            lam = lambda x, name: x.assign(**{name: np.arange(x.shape[0])})
            
            # org: rn_org <-> g_id <-> wgn
            org_order = (self.add_row_number(name = rn_org)
                             .add_group_number(name = g_id, by = by)
                             .select([rn_org, g_id])
                             .add_row_number(name = wgn, by = g_id)
                             .to_pandas(copy = False)
                             )
                               
            # arranged: rn_arr <-> g_id <-> wgn
            arranged_order = (self.add_group_number(name = g_id, by = by)
                                  .add_row_number(name = rn_arr)
                                  .to_pandas(copy = False)
                                  .groupby(g_id, sort = False)
                                  .apply(pdf_sortit)
                                  .reset_index(drop = True)
                                  .loc[:, [g_id, rn_arr]]
                                  .groupby(g_id, sort = False)
                                  .apply(lambda x: lam(x, wgn))
                                  .reset_index(drop = True)
                                  .loc[:, [g_id, wgn, rn_arr]]
                                  )
            
            # rn_org <-> rn_arr                   
            order_df = (org_order.merge(arranged_order, on = [g_id, wgn])
                                 .loc[:, [rn_org, rn_arr]]
                                 .sort_values(rn_org)
                                 )
            
            res = (self.__data
                       .iloc[list(order_df[rn_arr]), :]
                       .reset_index(drop = True)
                       )
            
        return tidyframe(res, check = False, copy = False)
    
    ##########################################################################
    # filter
    ##########################################################################
        
    def filter(self, query = None, mask = None, by = None, **kwargs):
        '''
        subset some rows
        
        Parameters
        ----------
        query: str or a function
            when str, query string parsable by pandas eval
            when function, should output an array of booleans
            of length equal to numbers of rows of the input dataframe
        mask: list of booleans or
              numpy array or pandas Series of booleans
              of length equal to numbers of rows of the input dataframe
        by: Optional, str or list of strings
            column names to group by 
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Exactly one arg among 'query' and 'mask' should be provided
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # query with pandas eval. pandas eval does not support complicated expressions.
        >>> penguins_tidy.filter("body_mass_g > 4000")
        >>> 
        >>> # subset with a mask -- list or array or pandas Series of precomputed booleans
        >>> penguins_tidy.filter(mask = (penguins_tidy.pull("year") == 2007))
        >>> # equivalently:
        >>> # penguins_tidy.filter(lambda x: x.year == 2007)
        >>> 
        >>> # use complex expressions as a lambda function and filter
        >>> penguins_tidy.filter(lambda x: x['bill_length_mm'] > np.mean(x['bill_length_mm']))
        >>> 
        >>> # per group filter retains the row order
        >>> penguins_tidy.filter(lambda x: x['bill_length_mm'] > np.mean(x['bill_length_mm']), by = 'sex')
        >>> 
        >>> # using kwargs
        >>> penguins_tidy.filter(lambda x, **kwargs: x.year == kwargs['some_kwarg'],
        >>>                      some_kwarg = 2009
        >>>                      )
        '''
        if query is None and mask is None:
            raise Exception("Both 'query' and 'mask' cannot be None")
        if query is not None and mask is not None:
            raise Exception("One among 'query' and 'mask' should be None")
        
        # validate query
        if query is not None:
            assert callable(query) or isinstance(query, str)
            
        # validate mask
        if mask is not None:
            assert isinstance(mask, (pd.Series, np.array, list))
            
        # validate 'by'
        if by is not None:
            by = _enlist(by)
            self._validate_by(by)
        
        def handle_mask(ser):
            ser = ser.tolist()
            res = [False if pd.isna(x) else x for x in ser]
            return np.array(res)
        
        if query is not None and mask is None:
            if by is None:
                if isinstance(query, str):
                    res = self.__data.query(query)
                else:
                    if _is_kwargable(query):
                        res = (self.__data
                                   .assign(**{"__mask": lambda x: handle_mask(query(x, **kwargs))})
                                   .query("__mask")
                                   .drop(columns = "__mask")
                                   )
                    else:
                        res = (self.__data
                               .assign(**{"__mask": lambda x: handle_mask(query(x))})
                               .query("__mask")
                               .drop(columns = "__mask")
                               )
            else: # grouped case
                ro_name = _generate_new_string(self.colnames)
                if isinstance(query, str):
                    res = (self.group_modify(lambda chunk: chunk.query(query)
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             , row_order_column_name = ro_name
                                             )
                               )
                else:
                    if _is_kwargable(query):
                        res = (self.group_modify(lambda chunk: (
                                                     chunk.assign(**{"__mask": lambda x: handle_mask(query(x, **kwargs))})
                                                          .query("__mask")
                                                          .drop(columns = "__mask")
                                                          )
                                                 , by = by
                                                 , is_pandas_udf = True
                                                 , preserve_row_order = True
                                                 , row_order_column_name = ro_name
                                                 )
                                   )
                    else:
                        res = (self.group_modify(lambda chunk: (
                                                     chunk.assign(**{"__mask": lambda x: handle_mask(query(x))})
                                                          .query("__mask")
                                                          .drop(columns = "__mask")
                                                          )
                                                 , by = by
                                                 , is_pandas_udf = True
                                                 , preserve_row_order = True
                                                 , row_order_column_name = ro_name
                                                 )
                                   )
                res = res.relocate(self.colnames)
        
        if isinstance(res, pd.DataFrame):
            if query is None and mask is not None:
                res = self.__data.loc[mask, :]
            
            res = res.reset_index(drop = True)     
            return tidyframe(res, check = False)
        else:
            return res
    
    ##########################################################################
    # distinct
    ##########################################################################
        
    def distinct(self, column_names = None, keep_all = False):
        '''
        subset unique rows from the dataframe
        
        Parameters
        ----------
        column_names: string or a list of strings
            Names of the column for distinct
        keep_all: bool (default: False)
            Whether to keep all the columns or only the 'column_names'
        by: Optional, string or a list of strings
            Column names to groupby
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. distinct preserves the order of the rows of the input dataframe.
        2. 'column_names' and 'by' should not have common column names.
        3. When keep_all is True, first rows corresponding a unique combination
           is preserved.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> tidy_penguins.distinct() # distinct over all columns
        >>> 
        >>> tidy_penguins.distinct('species') # keep only 'distinct' columns
        >>> tidy_penguins.distinct('species', keep_all = True) # keep all columns
        >>> 
        >>> tidy_penguins.distinct(['bill_length_mm', 'bill_depth_mm'])
        '''
        
        cn = self.colnames
        if column_names is not None:
            assert _is_string_or_string_list(column_names),\
                "arg 'column_names' should be a string or a list of strings"
            column_names = list(setlist(_enlist(column_names)))
            assert set(column_names).issubset(cn),\
                "arg 'column_names' should contain valid column names"
        else:
            column_names = cn
        assert isinstance(keep_all, bool)
        
        res = (self.__data
                   .drop_duplicates(subset = column_names
                                    , keep = "first"
                                    , ignore_index = True
                                    )
                   )
        
        if not keep_all:
            res = res.loc[:, column_names]
                       
        return tidyframe(res, check = False)
    
    ##########################################################################
    # mutate
    ##########################################################################
    
    def _mutate(self, dictionary, order_by = None, **kwargs):
        
        nr = self.nrow
        cn = self.colnames
        
        # strategy
        # created 'mutated' copy and keep adding/modifying columns
        
        keys = dictionary.keys()
        # ensure new column names to be created are valid (do not start with '_')
        keys_flattened = list(
          np.concatenate([[x] if np.isscalar(x) else list(x) for x in keys])
          )
        assert np.all([_is_valid_colname(x) for x in keys_flattened]),\
          (f"column names to be created/modified should be valid column names. "
           "A valid column name should be a string not starting from '_'"
           )
        
        # handle 'order_by':
        # Orders the dataframe for 'mutate' and keeps a row order column
        # at the end of mutate operation, the dataframe is sorted in original 
        # order and row order column is deleted
        rn_name = _generate_new_string(list(set(cn + keys_flattened)))
        if order_by is not None:
            order_by = self._clean_order_by(order_by)
            mutated = (self.__data
                           .copy()
                           .assign(**{rn_name: lambda x: np.arange(x.shape[0])})
                           .sort_values(by = [x[0] for x in order_by],
                                        ascending = [x[1] == "asc" for x in order_by]
                                        )
                           )
        else:
            mutated = self.__data.copy()
        
        keys = dictionary.keys()
        # ensure new column names to be created are valid (do not start with '_')
        keys_flattened = list(
          np.concatenate([[x] if np.isscalar(x) else list(x) for x in keys])
          )
        assert np.all([_is_valid_colname(x) for x in keys_flattened]),\
          (f"column names to be created/modified should be valid column names. "
           "A valid column name should be a string not starting from '_'"
           )
        
        for akey in dictionary:
            
            assert isinstance(akey, (str, tuple)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            assert _is_string_or_string_list(list(akey)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            
            rhs = dictionary[akey]
            
            # akey is a string or a list of strings
            # no longer a tuple
            if isinstance(akey, tuple):
                if len(akey) == 1:
                    akey = akey[0]
                else:
                    assert len(akey) == len(set(akey)),\
                        "LHS should have unique strings"
                    akey = list(akey)
         
            if isinstance(akey, str):
                # three cases:
                #   1. direct assign
                #   2. assign via function
                #   3. assign via simple function
                #       3a. assign a preexisting column
                #       3b. assign with column args
                
                # 1. direct assign
                if isinstance(rhs, ((pd.Series, np.ndarray)) 
                                    or np.isscalar(rhs)):
                    mutated[akey] = rhs
                
                # 2. assign via function -- pandas style UDF
                elif callable(rhs) or isinstance(rhs, str):
                    if callable(rhs):
                        if _is_kwargable(rhs):
                            mutated[akey] = rhs(mutated, **kwargs)
                        else:
                            mutated[akey] = rhs(mutated)
                    else:
                        rhs = eval("lambda x, **kwargs: " + rhs)
                        mutated[akey] = rhs(mutated)
                                 
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) > 0 and len(rhs) <= 2,\
                        (f"When RHS is a tuple, RHS should not have more than "
                         "two elements"
                         )
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    # 3a. assign a preexisting column
                    if len(rhs) == 1:
                        if callable(rhs[0]):
                            assert akey in list(mutated.columns),\
                                (f"When RHS is a tuple with function alone, "
                                 "{akey} should be an exisiting column"
                                 )
                            if callable(rhs[0]):
                                if _is_kwargable(rhs[0]):
                                    mutated[akey] = rhs[0](mutated[akey], **kwargs)
                                else:
                                    mutated[akey] = rhs[0](mutated[akey])
                        else:
                            # string case
                            fun = eval("lambda x, **kwargs: " + rhs[0])
                            mutated[akey] = fun(mutated[akey], **kwargs)
                    # 3b. assign with column args
                    else:
                        assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS tuple has two elements, the second "
                             "element should be a string or a list of strings"
                             )
                        cols = _enlist(rhs[1])
                        assert set(cols).issubset(list(mutated.columns)),\
                            f"RHS should contain valid columns for LHS '{akey}'"
                        
                        if callable(rhs[0]):
                            if _is_kwargable(rhs[0]):
                                mutated[akey] = rhs[0](*[mutated[acol] for acol in cols], **kwargs)
                            else:
                                mutated[akey] = rhs[0](*[mutated[acol] for acol in cols])
                        else:
                            # string case
                            chars = list(string.ascii_lowercase[-3:] 
                                         + string.ascii_lowercase[:-3]
                                         )[:len(cols)]
                            fun = eval(("lambda " 
                                        + ", ".join(chars) 
                                        + ", **kwargs: " 
                                        + rhs[0]
                                        )
                                       )
                            mutated[akey] = (
                                fun(*[mutated[acol] for acol in cols], **kwargs)
                                )
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
            
            # multiple assignments
            else:
                # 1. direct assign is not supported for multi assignment
                
                # 2. assign via function
                if callable(rhs) or isinstance(rhs, str):
                    
                    if callable(rhs):
                        if _is_kwargable(rhs):
                            rhs_res = rhs(mutated, **kwargs)
                        else:
                            rhs_res = rhs(mutated)
                    else:
                        rhs = eval("lambda x, **kwargs: " + rhs)
                        rhs_res = rhs(mutated, **kwargs)
                        
                    assert (isinstance(rhs_res, list) 
                            and len(rhs_res) == len(akey)
                            ),\
                        ("RHS should output a list of length equal to "
                         "length of LHS for key: {akey}"
                         )
                         
                    for apair in zip(akey, rhs_res):
                        mutated[apair[0]] = apair[1]
                        
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) > 0 and len(rhs) <= 2,\
                        (f"When RHS is a tuple, RHS should not have more "
                         "than two elements"
                         )
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    # 3a. assign a preexisting columns
                    if len(rhs) == 1:
                        assert set(akey).issubset(list(mutated.columns)),\
                            (f"When RHS is a tuple with function alone, {akey}"
                             "should be an exisiting columns"
                             )
                        if callable(rhs[0]):
                            if _is_kwargable(rhs[0]):
                                rhs_res = rhs[0](*[mutated[acol] for acol in akey]
                                                 , **kwargs
                                                 )
                            else:
                                rhs_res = rhs[0](*[mutated[acol] for acol in akey])
                        else:
                            chars = list(string.ascii_lowercase[-3:] 
                                         + string.ascii_lowercase[:-3]
                                         )[:len(akey)]
                            fun = eval(("lambda " 
                                        + ", ".join(chars) 
                                        + ", **kwargs: " 
                                        + rhs[0]
                                        )
                                       )  
                            rhs_res = fun(*[mutated[acol] for acol in akey], **kwargs)
                            
                        assert (isinstance(rhs_res, list)
                                and len(rhs_res) == len(akey)
                                ),\
                            (f"RHS should output a list of length equal "
                             "to length of LHS for key: {akey}"
                             )
                        for apair in zip(akey, rhs_res):
                            mutated[apair[0]] = apair[1]
                    # 3b. assign with column args
                    else:
                        assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS tuple has two elements, the second "
                             "element should be a string or a list of strings"
                             )
                        cols = _enlist(rhs[1])
                        assert set(cols).issubset(list(mutated.columns)),\
                            f"RHS should contain valid columns for LHS '{akey}'"
                        
                        if callable(rhs[0]):
                            if _is_kwargable(rhs[0]):
                                rhs_res = rhs[0](*[mutated[x] for x in cols], **kwargs)
                            else:
                                rhs_res = rhs[0](*[mutated[x] for x in cols])
                        else:
                            chars = list(string.ascii_lowercase[-3:] 
                                         + string.ascii_lowercase[:-3]
                                         )[:len(cols)]
                            fun = eval(("lambda " 
                                        + ", ".join(chars) 
                                        + ", **kwargs: " 
                                        + rhs[0]
                                        )
                                       )  
                            rhs_res = fun(*[mutated[x] for x in cols], **kwargs)
                        
                        assert (isinstance(rhs_res, list) 
                                and len(rhs_res) == len(akey)
                                ),\
                            ("RHS should output a list of length equal "
                             "to length of LHS for key: {akey}"
                             )
                        for apair in zip(akey, rhs_res):
                            mutated[apair[0]] = apair[1]
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
        
        if order_by is not None:
            mutated = mutated.sort_values(rn_name).drop(columns = [rn_name])
              
        return tidyframe(mutated
                          , check = False
                          , copy = False
                          )

    # mutate_across    
    def _mutate_across(self
                       , func
                       , column_names = None
                       , predicate = None
                       , prefix = ""
                       , order_by = None
                       , **kwargs
                       ):
        
        cn = self.colnames
        assert callable(func) or isinstance(func, str),\
            ("arg 'func' should be a function or a string which is convertible "
             "to a lambda function"
             )
        assert isinstance(prefix, str)

        if order_by is not None:
            order_by = self._clean_order_by(order_by)
            cn_prefix = [x + prefix for x in cn]
            rn_name = _generate_new_string(list(set(cn + cn_prefix)))
            mutated = (self.__data
                           .copy()
                           .assign(**{rn_name: lambda x: np.arange(x.shape[0])})
                           .sort_values(by = [x[0] for x in order_by],
                                        ascending = [x[1] == "asc" for x in order_by]
                                        )
                           )
        else:
            mutated = self.__data.copy()
        
        if (column_names is not None) and (predicate is not None):
            raise Exception(("Exactly one among 'column_names' and 'predicate' "
                             "should be None")
                             )
        
        if (column_names is None) and (predicate is None):
            raise Exception("Exactly one among 'column_names' and 'predicate' "
                            "should be None"
                            )
        
        # use column_names
        if column_names is not None:
            assert isinstance(column_names, list)
            assert all([isinstance(acol, str) for acol in column_names])
        # use predicate to assign appropriate column_names
        else:
            mask = list(self.__data.apply(predicate, axis = 0))
            assert all([isinstance(x, bool) for x in mask])(self.group_modify(lambda chunk: chunk.query(query)
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             , row_order_column_name = ro_name
                                             )
                               )
            column_names = self.__data.columns[mask]
        
        # make a copy of the dataframe and apply mutate in order
        for acol in column_names:
            if callable(func):
                if _is_kwargable(func):
                    mutated[prefix + acol] = func(mutated[acol], **kwargs)
                else:
                    mutated[prefix + acol] = func(mutated[acol])
            else:    
                func = eval("lambda x, **kwargs: " + func)
                mutated[prefix + acol] = func(mutated[acol], **kwargs)
        
        if order_by is not None:
            mutated = mutated.sort_values(rn_name).drop(columns = [rn_name])
            
        return tidyframe(mutated, check = False, copy = False)
    
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
        '''
        Add or modify some columns
        
        Parameters
        ----------
        dictionary: dict
            A dictionary of mutating operations. This supports multiple 
            styles, see the examples
        func: callable
            A function to apply on the 'column_names' or the columns chosen by
            the 'predicate'.
        column_names: string or a list of strings
            Names of the columns to apply 'func' on
        predicate: callable
            A function to choose columns. Input: pandas series, output: bool
        prefix: string
            Prefix the resulting summarization column after applying 'func'
        by: string or a list of strings
            column names to group by
        order_by: string or a list of strings
            Names of columns to order by, before applying the mutating function
        **kwargs
            for lambda functions in dictionary or func
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. mutate works in two modes:
            a. direct assign: A series or 1-D array is assignged as a column
            b. dictionary mode: A key should be a resulting column name. Value
                should be a specification of the mutating operation. See the 
                examples for various sytles.
            c. across mode: A function is applied across a set of columns or 
                columns selected by a predicate function.
        2. mutate preserves row order.
        3. 'by'(groupby) and 'order_by' columns should not be mutated.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # mutate with dict
        >>> penguins_tidy.mutate({
        >>>   # 1. direct assign
        >>>   'ind' : np.arange(344), 
        >>>   
        >>>   # 2. using pandas style lambda function
        >>>   "yp1": lambda x: x['year'] + 1,
        >>>   
        >>>   # 2a. pass the content of lambda function as a string
        >>>   "yp1_string": "x['year'] + 1",
        >>>   
        >>>   # 3. pass a tuple of function and column list
        >>>   "yp1_abstract": (lambda x: x + 1, "year"),
        >>>   
        >>>   # 3a. pass the tuple of function content as string and the column list
        >>>   "yp2_abstract": ("x + 1", "year"),
        >>>   
        >>>   # 4. pass multiple columns
        >>>   "x_plus_y": ("x + y", ["bill_length_mm", "bill_depth_mm"]),
        >>>   # the above is equivalent to:
        >>>   # "x_plus_y": (lambda x, y: x + y, ["bill_length_mm", "bill_depth_mm"]),
        >>>   
        >>>   # 5. output multiple columns as a list
        >>>   ('x', 'y'): lambda x: [x['year'] - 1, x['year'] + 1],
        >>>   # the above is equivalent to:
        >>>   # ('x2', 'y2'): "[x['year'] - 1, x['year'] + 1]",
        >>>   
        >>>   # change an existing column: add one to 'bill_length_mm'
        >>>   'bill_length_mm': ("x + 1", )
        >>>   # the above is equivalent to these:
        >>>   # 'bill_length_mm': ("x + 1", 'bill_length_mm'),
        >>>   # 'bill_length_mm': (lambda x: x + 1, 'bill_length_mm')
        >>>   })
        >>>     
        >>> # mutate with dict and groupby    
        >>> penguins_tidy.mutate({'x' : "x['year'] + np.mean(x['year']) - 4015"}
        >>>                      , by = 'sex'
        >>>                      )
        >>> 
        >>> # mutate can use columns created in the dict before
        >>> (penguins_tidy.select('year')
        >>>               .mutate({'yp1': ("x + 1", 'year'),
        >>>                        'yp1m1': ("x - 1", 'yp1')
        >>>                       })
        >>>               )
        >>>                         
        >>> # use kwargs
        >>> (penguins_tidy
        >>>  .select('year')
        >>>  .mutate({'yp1': ("x + kwargs['akwarg']", 'year')}, akwarg = 10)
        >>>  )
        >>> 
        >>> # 'order_by' some column before the mutate opeation
        >>> # order_by column 'bill_length_mm' before computing cumsum over 'year' columns
        >>> # row order is preserved
        >>> cumsum_df = (penguins_tidy.select(['year', 'species', 'bill_length_mm'])
        >>>                           .mutate({'year_cumsum': (np.cumsum, 'year')},
        >>>                                   order_by = 'bill_length_mm'
        >>>                                   )
        >>>                           )
        >>> cumsum_df
        >>> # confirm the computation:
        >>> cumsum_df.arrange('bill_length_mm')
        >>> 
        >>> # across mode with column names
        >>> (penguins_tidy.select(['bill_length_mm', 'body_mass_g'])
        >>>               .mutate(column_names = ['bill_length_mm', 'body_mass_g']
        >>>                       , func = lambda x: x - np.mean(x)
        >>>                       , prefix = "demean_"
        >>>                       )
        >>>               )
        >>>               
        >>> # grouped across with column names
        >>> (penguins_tidy.select(['bill_length_mm', 'body_mass_g', 'species'])
        >>>               .mutate(column_names = ['bill_length_mm', 'body_mass_g'],
        >>>                       func = lambda x: x - np.mean(x),
        >>>                       prefix = "demean_",
        >>>                       by = 'species'
        >>>                       )
        >>>               )
        >>>   
        >>> # across mode with predicate
        >>> penguins_tidy.mutate(func = lambda x: x - np.mean(x),
        >>>                      predicate = dtypes.is_numeric_dtype,
        >>>                      prefix = "demean_"
        >>>                      )
        >>>   
        >>> # grouped across with predicate without prefix
        >>> # this will return a copy with columns changed without changing names
        >>> penguins_tidy.mutate(func = lambda x: x - np.mean(x),
        >>>                      predicate = dtypes.is_numeric_dtype,
        >>>                      by = 'species'
        >>>                      )
        '''
        if dictionary is None and func is None:
            raise Exception(("Either dictionary or func with "
                             "predicate/column_names should be provided."
                            ))
        if by is None:
            if dictionary is not None:
                res = self._mutate(dictionary, order_by = order_by, **kwargs)
            else:
                res = self._mutate_across(func
                                          , column_names = column_names
                                          , predicate = predicate
                                          , prefix = prefix
                                          , order_by = order_by
                                          , **kwargs
                                          )
        else:
            self._validate_by(by)
            by = _enlist(by)
            cn = self.colnames
            
            if dictionary is not None:
                
                keys = dictionary.keys()
                keys_flattened = list(
                  np.concatenate(
                    [[x] if np.isscalar(x) else list(x) for x in keys]
                    )
                  )
                # keys should not intersect with 'by' columns
                assert len(set(by).intersection(keys_flattened)) == 0,\
                    ("column names to be created by 'mutate' should not "
                     "intersect with 'by' columns"
                     )
                res = self.group_modify(
                    func = lambda chunk: chunk._mutate(dictionary, order_by = order_by)
                    , by = by
                    , preserve_row_order = True
                    , row_order_column_name = _generate_new_string(cn)
                    , is_pandas_udf = False
                    , **kwargs
                    )
                # set the column order
                # original columns in old order
                # new column come next in dict order
                col_order = cn + list(setlist(keys_flattened).difference(cn))
                res = res.select(col_order)
            else:
                res = self.group_modify(
                    func = lambda chunk: (
                        chunk._mutate_across(func
                                             , column_names = column_names
                                             , predicate = predicate
                                             , prefix = prefix
                                             , order_by = order_by
                                             )
                        )
                    , by = by
                    , preserve_row_order = True
                    , row_order_column_name = _generate_new_string(cn)
                    , is_pandas_udf = False
                    , **kwargs
                    )
                # set the column order
                col_order = res.colnames
                col_order = cn + list(setlist(col_order).difference(cn))
                res = res.select(col_order)

        return res
    
    
    ##########################################################################
    # summarise
    ##########################################################################
 
    def _summarise(self, dictionary, **kwargs):
        
        nr = self.nrow
        cn = self.colnames
        summary_dict = {}
        
        def _validate_rhs_val(akey, rhs_val):
            if not pd.isna(rhs_val):
                if not np.isscalar(rhs_val):
                    if not (len(rhs_val) == 1 and np.isscalar(rhs_val[0])):
                        raise Exception((f"Summarised value for key {akey} does not"
                                         " turn out to be a scalar or cannot be "
                                         "converted to a scalar")
                                        )
            return None
        
        for akey in dictionary:
            
            assert isinstance(akey, (str, tuple)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            assert _is_string_or_string_list(list(akey)),\
                "LHS (dict keys) should be a string or a tuple of strings"
            
            rhs = dictionary[akey]
            
            # akey is a string or a list of strings
            # no longer a tuple
            if isinstance(akey, tuple):
                if len(akey) == 1:
                    akey = akey[0]
                else:
                    assert len(akey) == len(set(akey)),\
                        "LHS should have unique strings"
                    akey = list(akey)
         
            if isinstance(akey, str):
                # two cases:
                #   1. assign via function
                #   2. assign with column args
                
                # 1. assign via function
                if callable(rhs) or isinstance(rhs, str):
                    if callable(rhs):
                        if _is_kwargable(rhs):
                            rhs_val = rhs(self.__data, **kwargs)
                        else:
                            rhs_val = rhs(self.__data)
                            
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                    else:
                        rhs = eval("lambda x, **kwargs: " + rhs)
                        rhs_val = rhs(self.__data, **kwargs)
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                                 
                # 2. assign via simple function
                elif isinstance(rhs, tuple):
                    if len(rhs) == 1:
                        rhs = (rhs[0], akey)
                    assert len(rhs) == 2,\
                        f"When RHS is a tuple, RHS should have two elements"
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS is a tuple, the second "
                             "element should be a string or a list of strings"
                             )
                    cols = _enlist(rhs[1])
                    assert set(cols).issubset(cn),\
                        f"RHS should contain valid columns for LHS '{akey}'"
                    
                    if callable(rhs[0]):
                        if _is_kwargable(rhs[0]):
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols]
                                             , **kwargs
                                             )
                        else:
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols])
                            
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                    else:
                        # string case
                        chars = list(string.ascii_lowercase[-3:] 
                                     + string.ascii_lowercase[:-3]
                                     )[:len(cols)]
                        fun = eval(("lambda " 
                                    + ", ".join(chars) 
                                    + ", **kwargs: " 
                                    + rhs[0]
                                    )
                                   )
                      
                        rhs_val = fun(*[self.__data[acol] for acol in cols], **kwargs)
                        _validate_rhs_val(akey, rhs_val)
                        summary_dict[akey] = rhs_val
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
            
            # multiple assignments
            else:
                # 1. assign via function
                if callable(rhs) or isinstance(rhs, str):
                    if not callable(rhs):
                        rhs = eval("lambda x, **kwargs: " + rhs)
                    
                    if _is_kwargable(rhs):
                        rhs_val = rhs(self.__data, **kwargs)
                    else:
                        rhs_val = rhs(self.__data)
                        
                    [_validate_rhs_val(akey, x) for x in rhs_val]
                        
                    assert (isinstance(rhs_val, list) 
                            and len(rhs_val) == len(akey)
                            ),\
                        (f"RHS should output a list of length equal to "
                         "length of LHS for key: {akey}"
                         )
                         
                    for i in range(len(akey)):
                      summary_dict[akey[i]] = rhs_val[i]
                        
                # 3. assign via simple function
                elif isinstance(rhs, tuple):
                    assert len(rhs) == 2,\
                        f"When RHS is a tuple, RHS should have two elements"
                    assert callable(rhs[0]) or isinstance(rhs[0], str),\
                        (f"When RHS is a tuple, first element of RHS should be "
                         "a function or a string which can be enclosed in a "
                         "lambda function and evaluated"
                         )
                    assert _is_string_or_string_list(rhs[1]),\
                            (f"When RHS is a tuple, the second "
                             "element should be a string or a list of strings"
                             )
                    cols = _enlist(rhs[1])
                    assert set(cols).issubset(cn),\
                        f"RHS should contain valid columns for LHS '{akey}'"
                    
                    if callable(rhs[0]):
                        if _is_kwargable(rhs[0]):
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols]
                                             , **kwargs
                                             )
                        else:
                            rhs_val = rhs[0](*[self.__data[acol] for acol in cols])
                    else:
                        chars = list(string.ascii_lowercase[-3:] 
                                     + string.ascii_lowercase[:-3]
                                     )[:len(cols)]
                        fun = eval(("lambda " 
                                    + ", ".join(chars) 
                                    + ", **kwargs: " 
                                    + rhs[0]
                                    )
                                   )  
                        rhs_val = fun(*[self.__data[acol] for acol in cols], **kwargs)
                        
                    [_validate_rhs_val(akey, x) for x in rhs_val]
                    
                    assert (isinstance(rhs_val, list) 
                            and len(rhs_val) == len(akey)
                            ),\
                        (f"RHS should output a list of length equal "
                         "to length of LHS for key: {akey}"
                         )
                         
                    for i in range(len(akey)):
                      summary_dict[akey[i]] = rhs_val[i]
                else:
                    raise Exception((f"RHS for key '{akey}' should be in some "
                                     "standard form"
                                     ))
                                     
        res = pd.DataFrame(summary_dict, index = [0])
        return tidyframe(res, copy = False, check = False)
  
    def summarise(self
                  , dictionary = None
                  , func = None
                  , column_names = None
                  , predicate = None
                  , prefix = ""
                  , by = None
                  , **kwargs
                  ):
        '''
        Creates one row per group summarising the input dataframe
        
        Parameters
        ----------
        dictionary: dict
            A dictionary of summarising operations. This supports multiple 
            styles, see the examples
        func: callable
            A function to apply on the 'column_names' or the columns chosen by
            the 'predicate'. func should return a scalar.
        column_names: string or a list of strings
            Names of the columns to apply 'func' on
        predicate: callable
            A function to choose columns. Input: pandas series, output: bool
        prefix: string
            Prefix the resulting summarization column after applying 'func'
        by: string or a list of strings
            column names to group by
        **kwargs
            for lambda functions in dictionary or func
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. When by is None, summarise outputs a single row dataframe.
        2. summarise function works in two modes:
            a. dictionary mode: A key should be a resulting column name. Value
                should be a specification of the summarising operation. See the 
                examples for various sytles.
            b. across mode: A function is applied across a set of columns or 
                columns selected by a predicate function.
        
        Examples
        --------
        >>> import pandas.api.types as dtypes
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # summarise in dict mode
        >>> penguins_tidy.summarise({
        >>>     # using pandas style lambda function
        >>>     "a_mean": lambda x: x['year'].mean(),
        >>>     
        >>>     # pass the content of lambda function as a string
        >>>     "b_mean": "x['year'].mean()",
        >>>     
        >>>     # pass a tuple of function and column list
        >>>     "a_median": (np.median, "year"),
        >>>     
        >>>     # pass a tuple of function to retain same column name
        >>>     "year": (np.median, ),
        >>>     
        >>>     # pass multiple columns as a string
        >>>     "x_plus_y_mean": ("np.mean(x + y)", ["bill_length_mm", "bill_depth_mm"]),
        >>>     
        >>>     # output multiple columns as a list (pandas style)
        >>>     ('x', 'y'): lambda x: list(np.quantile(x['year'], [0.25, 0.75])),
        >>>     
        >>>     # same as above in string style
        >>>     ('x2', 'y2'): "list(np.quantile(x['year'], [0.25, 0.75]))",
        >>>     
        >>>     # tuple style with multiple output
        >>>     ('A', 'B'): ("[np.mean(x + y), np.mean(x - y)]"
        >>>                  , ["bill_length_mm", "bill_depth_mm"]
        >>>                  )
        >>>     })
        >>>     
        >>> # grouped summarise in dict mode
        >>> penguins_tidy.summarise({"a_mean": (np.mean, 'year')},
        >>>                         by = ['species', 'sex']
        >>>                         )
        >>>                         
        >>> # use kwargs
        >>> penguins_tidy.summarise(
        >>>   {"a_mean": lambda x, **kwargs: x['year'].mean() + kwargs['leap']},
        >>>   by = ['species', 'sex'],
        >>>   leap = 4
        >>>   )
        >>> 
        >>> # equivalently:
        >>> penguins_tidy.summarise(
        >>>   {"a_mean": "x['year'].mean() + kwargs['leap']"},
        >>>   by = ['species', 'sex'],
        >>>   leap = 4
        >>>   )
        >>> 
        >>> # across mode with column names
        >>> penguins_tidy.summarise(
        >>>   func = np.mean,
        >>>   column_names = ['bill_length_mm', 'bill_depth_mm']
        >>>   )
        >>>   
        >>> # across mode with predicate
        >>> penguins_tidy.summarise(
        >>>   func = np.mean,
        >>>   predicate = dtypes.is_numeric_dtype,
        >>>   prefix = "avg_"
        >>>   )
        >>>   
        >>> # grouped across with predicate
        >>> penguins_tidy.summarise(
        >>>   func = np.mean,
        >>>   predicate = dtypes.is_numeric_dtype,
        >>>   prefix = "avg_",
        >>>   by = ['species', 'sex']
        >>>   )
        '''
        
        if dictionary is None and func is None:
            raise Exception(("Either dictionary or func with "
                             "predicate/column_names should be provided"
                            ))
        
        # assertions for across case                    
        if dictionary is None:
            assert callable(func) or isinstance(func, str),\
                ("arg 'func' should be a function or a string which is convertible "
                 "to a lambda function"
                 )
            assert isinstance(prefix, str)
            if (column_names is not None) and (predicate is not None):
                raise Exception(("Exactly one among 'column_names' and 'predicate' "
                                 "should be None")
                                 )
            if (column_names is None) and (predicate is None):
                raise Exception("Exactly one among 'column_names' and 'predicate' "
                                "should be None"
                                )
            
            # use column_names
            if column_names is not None:
                self._validate_column_names(column_names)
                column_names = _enlist(column_names)
            # use predicate to assign appropriate column_names
            else:
                mask = list(self.__data.apply(predicate, axis = 0))
                assert all([isinstance(x, bool) for x in mask]),\
                    "predicate should return a boolean per column"
                column_names = self.__data.columns[mask]
            
            if isinstance(func, str):
                func = eval("lambda x, **kwargs: " + func)
            prefixed_names = [prefix + x for x in column_names]
            
        # simple summarise
        if by is None:
            if dictionary is not None:
                res = self._summarise(dictionary, **kwargs)
            else:
                res = (self.__data[column_names]
                           .agg(func, **kwargs)
                           .to_frame()
                           .T
                           .rename(columns = dict(zip(column_names, prefixed_names)))
                           )
                res = tidyframe(res, copy = False)
        else:
            self._validate_by(by)
            by = _enlist(by)
            cn = self.colnames
            
            if dictionary is not None:
                # agg column names should not intersect with by columns
                keys = dictionary.keys()
                keys_flattened = list(
                  np.concatenate(
                    [[x] if np.isscalar(x) else list(x) for x in keys]
                    )
                  )
                assert len(set(by).intersection(keys_flattened)) == 0,\
                    ("column names to be created by 'summarise' should not "
                     "intersect with 'by' columns"
                     )
                # special case to bypass _summarise when the agg is over a 
                # single column
                dv = dictionary.values() 
                special_case = False
                if all([isinstance(x, tuple) for x in dv]):
                    # cover these: 
                    # {'new_col': (np.mean, 'col')}
                    # {'col': (np.mean, )}
                    if all([len(x) == 1 or len(_enlist(x[1])) == 1 for x in dv]):
                        if all([callable(x[0]) for x in dv]):
                            special_case = True
                
                if special_case:
                    def create_summary(anitem):
                        # case: {'col': (np.mean, )}
                        if len(anitem[1]) == 1:
                            res = (self.__data
                                       .groupby(by, dropna = False, sort = False)[anitem[0]]
                                       .agg(anitem[1][0])
                                       .reset_index()
                                       )
                        else:
                        # case: {'new_col': (np.mean, 'col')}
                            res = (self.__data
                                       .groupby(by, dropna = False, sort = False)[anitem[1][1]]
                                       .agg(anitem[1][0])
                                       .reset_index()
                                       .rename(columns = {anitem[1][1]: anitem[0]})
                                       )
                        return res
                      
                    aggs = [create_summary(x) for x in dictionary.items()]
                    if len(aggs) == 1:
                        res = aggs[0]
                    else:
                        res = functools.reduce(lambda df1, df2: df1.merge(df2, on = by)
                                               , aggs
                                               )
                    res = tidyframe(res, check = False, copy = False)
                else: # regular case
                    lam = lambda chunk: chunk._summarise(dictionary, **kwargs)
                    res = self.group_modify(func = lam, by = by)
                    
                # set the column order
                # by columns come first
                # aggreagated columns come next
                col_order = by + keys_flattened
                res = res.select(col_order)
            else:
                rowid_name = _generate_new_string(self.colnames)
                group_mapper = (self.distinct(by)
                                    .add_row_number(rowid_name)
                                    .to_pandas()
                                    )
                res = (self.__data
                           .merge(group_mapper, on = by)
                           .drop(columns = by)
                           .groupby(rowid_name, dropna = False, sort = False)[column_names]
                           .agg(func, **kwargs)
                           .reset_index()
                           .merge(group_mapper, on = rowid_name)
                           .drop(columns = rowid_name)
                           .rename(columns = dict(zip(column_names, prefixed_names)))
                           .loc[:, by + prefixed_names]
                           )
                res = tidyframe(res, check = False, copy = False)          
                # column order is set with 'by' column in the begining

        return res
    
    summarize = summarise
    
    ##########################################################################
    # Joins
    ##########################################################################
    
    # validate join args
    def _validate_join(self
                       , y
                       , how
                       , on
                       , on_x
                       , on_y
                       , sort
                       , suffix_y
                       ):
                           
        assert isinstance(y, tidyframe),\
            "arg 'y' should be a tidy pandas dataframe"
        
        cn_x = self.colnames
        cn_y = y.colnames
        
        assert isinstance(how, str),\
            "arg 'how' should be a string"
        assert how in ['inner', 'outer', 'left', 'right'],\
            "arg 'how' should be one among: 'inner', 'outer', 'left', 'right'"
        
        if on is None:
            assert on_x is not None and on_y is not None,\
                ("When arg 'on' is None, " 
                 "both args'on_x' and 'on_y' should not be None"
                 )
            assert _is_string_or_string_list(on_x),\
                "arg 'on_x' should be a string or a list of strings"
            assert _is_string_or_string_list(on_y),\
                "arg 'on_y' should be a string or a list of strings"
            
            on_x = _enlist(on_x)
            on_y = _enlist(on_y)
            
            assert _is_unique_list(on_x),\
                "arg 'on_x' should not have duplicates"
            assert _is_unique_list(on_y),\
                "arg 'on_y' should not have duplicates"    
            assert set(on_x).issubset(cn_x),\
                "arg 'on_x' should be a subset of column names of x"
            assert set(on_y).issubset(cn_y),\
                "arg 'on_y' should be a subset of column names of y"
            assert len(on_x) == len(on_y),\
                "Lengths of arg 'on_x' and arg 'on_y' should match"
        else: # on is provided
            assert on_x is None and on_y is None,\
                ("When arg 'on' is not None, " 
                 "both args'on_x' and 'on_y' should be None"
                 )
            assert _is_string_or_string_list(on),\
                "arg 'on' should be a string or a list of strings"
            on = _enlist(on)
            assert _is_unique_list(on),\
                "arg 'on' should not have duplicates"
            assert set(on).issubset(cn_x),\
                "arg 'on' should be a subset of column names of x"
            assert set(on).issubset(cn_y),\
                "arg 'on' should be a subset of column names of y"
        
        assert isinstance(suffix_y, str),\
            "arg 'suffix_y' should be a string"
            
        assert suffix_y != "",\
            "arg 'suffix_y' should be a non-empty string"
            
        assert isinstance(sort, bool),\
            "arg 'sort' should be a boolean"
        
        return None
        
    # join methods
    def _join(self
              , y
              , how
              , on = None
              , on_x = None
              , on_y = None
              , sort = True
              , suffix_y = "_y"
              , validate = True
              ):
        
        if validate:               
            self._validate_join(y = y
                                , how = how
                                , on = on
                                , on_x = on_x
                                , on_y = on_y
                                , sort = sort
                                , suffix_y = suffix_y
                                )
        
        cn_x = self.colnames
        cn_y = y.colnames
        nr_x = self.nrow
        nr_y = y.nrow
        
        # degenerate case
        if on is None:
            on_x = _enlist(on_x)
            on_y = _enlist(on_y)
            if on_x == on_y:
                on = util_copy.copy(on_x)
                on_x = None
                on_y = None
        
        if on is None:
            on_x = _enlist(on_x)
            on_y = _enlist(on_y)
            new_colnames = cn_x + list(setlist(cn_y).difference(on_y))
        else:
            on = _enlist(on)
            new_colnames = cn_x + list(setlist(cn_y).difference(on))
        
        if sort:
            if on is not None:
                res = (self
                       .__data
                       .assign(**{"__rn_x": np.arange(nr_x)})
                       .merge(right = (y.to_pandas(copy = False)
                                        .assign(**{"__rn_y": nr_y})
                                        )
                              , how = how
                              , on = on
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       .sort_values(by = ["__rn_x", "__rn_y"]
                                    , ignore_index = True
                                    )
                       .drop(columns = ["__rn_x", "__rn_y"])
                       )
            else:
                res = (self
                       .__data
                       .assign(**{"__rn_x": np.arange(nr_x)})
                       .merge(right = (y.to_pandas(copy = False)
                                        .assign(**{"__rn_y": nr_y})
                                        )
                              , how = how
                              , left_on = on_x
                              , right_on = on_y
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       .sort_values(by = ["__rn_x", "__rn_y"]
                                    , ignore_index = True
                                    )
                       .drop(columns = ["__rn_x", "__rn_y"])
                       )
                # remove the right keys
                on_y_with_suffix = [x + suffix_y for x in on_y]
                right_keys = set(res.columns).intersection(on_y_with_suffix)
                res = res.drop(columns = right_keys)
        else:
            if on is not None:
                res = (self
                       .__data
                       .merge(right = y.to_pandas(copy = False)
                              , how = how
                              , on = on
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       )
            else:
                res = (self
                       .__data
                       .merge(right = y.to_pandas(copy = False)
                              , how = how
                              , left_on = on_x
                              , right_on = on_y
                              , left_index = False
                              , right_index = False
                              , sort = False
                              , suffixes = ["", suffix_y]
                              , copy = True
                              , indicator = False
                              , validate = None
                              )
                       )
                # remove the right keys
                on_y_with_suffix = [x + suffix_y for x in on_y]
                right_keys = set(res.columns).intersection(on_y_with_suffix)
                res = res.drop(columns = right_keys)
        
        res = (tidyframe(res
                         , check = False
                         , copy = False
                         )
                .relocate(cn_x)
                )
        
        return res
        
    def inner_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        Joins columns of y to self by matching rows
        Includes only matching keys
        
        Parameters
        ----------
        y: tidyframe
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
        >>>                                  .select(['species', 'bill_length_mm', 'island'])
        >>>                                  )
        >>> penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
        >>>                                  .select(['species', 'island', 'bill_depth_mm'])
        >>>                                  )
        >>>                                  
        >>> penguins_tidy_s1.inner_join(penguins_tidy_s2, on = 'island')
        '''
                       
        res = self._join(y = y
                        , how = "inner"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
    
    
    def outer_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        Joins columns of y to self by matching rows
        Includes all keys
        
        Parameters
        ----------
        y: tidyframe
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
        >>>                                  .select(['species', 'bill_length_mm', 'island'])
        >>>                                  )
        >>> penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
        >>>                                  .select(['species', 'island', 'bill_depth_mm'])
        >>>                                  )
        >>>                     
        >>> penguins_tidy_s1.outer_join(penguins_tidy_s2, on = 'island')
        '''               
        res = self._join(y = y
                        , how = "outer"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
    
    full_join = outer_join
        
    def left_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        Joins columns of y to self by matching rows
        Includes all keys in self
        
        Parameters
        ----------
        y: tidyframe
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
        >>>                                  .select(['species', 'bill_length_mm', 'island'])
        >>>                                  )
        >>> penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
        >>>                                  .select(['species', 'island', 'bill_depth_mm'])
        >>>                                  )
        >>>                                  
        >>> penguins_tidy_s1.left_join(penguins_tidy_s2, on = 'island')
        '''               
        res = self._join(y = y
                        , how = "left"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
        
    def right_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        Joins columns of y to self by matching rows
        Includes all keys in y
        
        Parameters
        ----------
        y: tidyframe
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
        >>>                                  .select(['species', 'bill_length_mm', 'island'])
        >>>                                  )
        >>> penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
        >>>                                  .select(['species', 'island', 'bill_depth_mm'])
        >>>                                  )
        >>>                                  
        >>> penguins_tidy_s1.right_join(penguins_tidy_s2, on = 'island')
        '''               
        res = self._join(y = y
                        , how = "right"
                        , on = on
                        , on_x = on_x
                        , on_y = on_y
                        , sort = sort
                        , suffix_y = suffix_y
                        )
        return res
    
    # row of A which match B
    def semi_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        Joins columns of y to self by matching rows
        Includes keys in self if present in y
        
        Parameters
        ----------
        y: tidyframe
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
        >>>                                  .select(['species', 'bill_length_mm', 'island'])
        >>>                                  )
        >>> penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
        >>>                                  .select(['species', 'island', 'bill_depth_mm'])
        >>>                                  )
        >>>                                  
        >>> penguins_tidy_s2.semi_join(penguins_tidy_s1, on = 'island')
        '''
        self._validate_join(y = y
                            , how = "inner" # this has no significance
                            , on = on
                            , on_x = on_x
                            , on_y = on_y
                            , sort = sort
                            , suffix_y = suffix_y
                            )
        
        if on is None:
            y2 = y.distinct(on_y)
        else:
            y2 = y.distinct(on)
        
        res = self._join(y = y2
                         , how = "inner"
                         , on = on
                         , on_x = on_x
                         , on_y = on_y
                         , sort = sort
                         , suffix_y = suffix_y
                         , validate = False
                         )
        return res
    
    # rows of A that do not match B
    def anti_join(self
                   , y
                   , on = None
                   , on_x = None
                   , on_y = None
                   , sort = True
                   , suffix_y = "_y"
                   ):
        '''
        Joins columns of y to self by matching rows
        Includes keys in self if not present in y
        
        Parameters
        ----------
        y: tidyframe
        on: string or a list of strings
            Common column names to match
        on_x:
            Column names of self to be matched with arg 'on_y'
        on_y:
            Column names of y to be matched with arg 'on_x'
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
        >>>                                  .select(['species', 'bill_length_mm', 'island'])
        >>>                                  )
        >>> penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
        >>>                                  .select(['species', 'island', 'bill_depth_mm'])
        >>>                                  )
        >>>                                  
        >>> penguins_tidy_s2.anti_join(penguins_tidy_s1, on = 'island')
        '''
        self._validate_join(y = y
                            , how = "inner" # not significant
                            , on = on
                            , on_x = on_x
                            , on_y = on_y
                            , sort = sort
                            , suffix_y = suffix_y
                            )
        
        string = _generate_new_string(y.colnames)
        
        if on is None:
            on_value = on_y
        else:
            on_value = on
            
        res =  (self._join(y.distinct(on_value).add_row_number(name = string)
                           , how = "left"
                           , on = on
                           , on_x = on_x
                           , on_y = on_y
                           , sort = sort
                           , suffix_y = suffix_y
                           , validate = False
                           )
                    .filter(lambda x: x[string].isna())
                    .select(string, include = False)
                    )
        return res
    
    # cross join
    def cross_join(self, y, sort = True, suffix_y = "_y"):
        '''
        Joins columns of y to self by matching rows
        Includes all cartersian product
        
        Parameters
        ----------
        y: tidyframe
        sort: bool
            Whether to sort by row order of self and row order of y
        suffix_y: string
            suffix to append the columns of y which have same names as self's 
            column names
          
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Column names of self will not have a suffix.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
        >>>                                  .select(['species', 'bill_length_mm', 'island'])
        >>>                                  )
        >>> penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
        >>>                                  .select(['species', 'island', 'bill_depth_mm'])
        >>>                                  )
        >>>                                  
        >>> penguins_tidy_s2.cross_join(penguins_tidy_s1)
        '''
        
        assert isinstance(y, tidyframe),\
          "arg 'y' should be a tidyframe"
          
        assert isinstance(sort, bool),\
          "arg 'sort' should be a bool"
        
        assert isinstance(suffix_y, str),\
          "arg 'suffix_y' should be a string"
        
        assert suffix_y != "",\
          "arg 'suffix_y' should not be an empty string"
        
        if sort:
            res = (pd.merge((self.__data
                                 .assign(**{"__rn_x": lambda x: np.arange(x.shape[0])})
                                 )
                            , (y.to_pandas(copy = False)
                                .assign(**{"__rn_y": lambda x: np.arange(x.shape[0])}))
                            , how = "cross"
                            , left_index = False
                            , right_index = False
                            , suffixes = ["", suffix_y]
                            )
                     .sort_values(by = ["__rn_x", "__rn_y"]
                                  , ignore_index = True
                                  )
                     .drop(columns = ["__rn_x", "__rn_y"])
                     )
        else:  
            res = pd.merge(self.__data
                           , y.to_pandas(copy = False)
                           , how = "cross"
                           , left_index = False
                           , right_index = False
                           , suffixes = ["", suffix_y]
                           )
        
        return tidyframe(res, check = False, copy = False)
    
    ##########################################################################
    # bind methods
    ##########################################################################  
      
    def cbind(self, y):
        '''
        bind columns of y to self
        
        Parameters
        ----------
        y: a tidyframe with same number of rows
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. The tidyframe to be binded should same same number of rows.
        2. Column names of the tidyframe should be different from self.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> (penguins_tidy.select(['species', 'island'])
        >>>               .cbind(penguins_tidy.select(['bill_length_mm', 'bill_depth_mm']))
        >>>               )
        '''
        assert isinstance(y, tidyframe)
        # column names should differ
        assert len(set(self.colnames).intersection(y.colnames)) == 0,\
            "Column names among the dataframes should not be common. Did you intend to `cross_join` instead of `cbind`?"
                # number of rows should match
        assert self.nrow == y.nrow,\
            "Both dataframes should have same number of rows"
            
        res = (pd.concat([self.__data, y.to_pandas(copy = False)]
                        , axis = 1
                        , ignore_index = False # not to loose column names
                        )
                 .reset_index(drop = True)
                 )
        return tidyframe(res)
    
    def rbind(self, y):
        '''
        bind rows of y to self
        
        Parameters
        ----------
        y: a tidyframe
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Result will have union of column names of self and y.
        2. Missing values are created when a column name is present in one
           dataframe and not in the other.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> (penguins_tidy.select(['species', 'island'])
        >>>               .rbind(penguins_tidy.select(['island', 'bill_length_mm', 'bill_depth_mm']))
        >>>               )
        '''
        res = (pd.concat([self.__data, y.to_pandas()]
                         , axis = 0
                         , ignore_index = True
                         )
                 )
        return tidyframe(res)
    
    ##########################################################################
    # count and add_count
    ##########################################################################
    
    def count(self
              , column_names = None
              , name = 'n'
              , decreasing = True
              , wt = None
              ):
        '''
        count rows by groups
        
        Parameters
        ----------
        column_names: None or string or a list of strings
            Column names to groupby before counting the rows. 
            When None, counts all rows.
        name: string (default: 'n')
            Column name of the resulting count column
        decreasing: bool (default is True)
            Whether to sort the result in descending order of the count column
        wt: None or string
            When a string, should be a column name with numeric dtype. 
            When wt is None, rows are counted.
            When wt is present, then wt column is summed up.
            
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # count rows
        >>> penguins_tidy.count()
        >>> 
        >>> # count number of rows of 'sex' column
        >>> # count column is always ordered in descending order unless decreasing is False
        >>> penguins_tidy.count('sex', name = "cnt")
        >>> penguins_tidy.count('sex', name = "cnt", decreasing = False)
        >>> 
        >>> # sum up a column (weighted sum of rows)
        >>> penguins_tidy.count(['sex', 'species'], wt = 'year')
        '''
        
        assert (column_names is None
                or _is_string_or_string_list(column_names)
                ),\
            "arg 'column_names' is either None or a list of strings"
        if column_names is not None:
            self._validate_column_names(column_names)
            column_names = _enlist(column_names)
        assert isinstance(name, str),\
            "arg 'name' should a string"
        assert isinstance(decreasing, bool),\
            "arg 'decreasing' should be a boolean"
        if wt is not None:
            assert isinstance(wt, str),\
                "arg 'wt' should be a string"
            assert wt in self.colnames,\
                f"'wt' column '{wt}' should be a valid column name"
            assert pd.api.types.is_numeric_dtype(self.pull(wt)),\
                f"'wt' column '{wt}' should be of numeric dtype"
        
        if decreasing:
          sort_str = "desc"
        else:
          sort_str = "asc"
        
        if column_names is not None:
            assert isinstance(name, str),\
                "arg 'name' should be a string"
            assert name not in column_names,\
                ("arg 'name' should not be an element of arg "
                 "'column_names'"
                 )
            
            if wt is None:
                res = (self.__data
                           .groupby(column_names
                                    , sort = False
                                    , dropna = False
                                    )
                           .size()
                           .reset_index(drop = False)
                           .rename(columns = {0: name})
                           .sort_values(by = name
                                        , axis         = 0
                                        , ascending    = (not decreasing)
                                        , inplace      = False
                                        , kind         = 'quicksort'
                                        , na_position  = 'first'
                                        , ignore_index = True
                                        )
                           )
                res = tidyframe(res, check = False, copy = False)
            else:
                res = (self.summarise(
                              {name: (np.sum, wt)}
                               , by = column_names
                               , wt = wt
                               )
                           .arrange([(name, sort_str)])
                           )
        else: # column_names is None
            if wt is None:
                res = pd.DataFrame({name : self.nrow}
                                   , index = [0]
                                   )
                res = tidyframe(res, check = False, copy = False)
            else:
                res = self.summarise({name: "np.sum(x[kwargs['wt']])"}
                                     , wt = wt
                                     )
              
        return res

    def add_count(self
                  , column_names = None
                  , name = 'n'
                  , wt = None
                  ):
        
        '''
        adds counts of rows by groups as a column
        
        Parameters
        ----------
        column_names: None or string or a list of strings
            Column names to groupby before counting the rows. 
            When None, counts all rows.
        name: string (default: 'n')
            Column name of the resulting count column
        wt: None or string
            When a string, should be a column name with numeric dtype. 
            When wt is None, rows are counted.
            When wt is present, then wt column is summed up.
            
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # count rows
        >>> penguins_tidy.add_count()
        >>> 
        >>> # count number of rows of 'sex' column
        >>> penguins_tidy.add_count('sex', name = "cnt")
        >>> 
        >>> # sum up a column (weighted sum of rows)
        >>> penguins_tidy.add_count(['sex', 'species'], wt = 'year')
        '''
        
        if column_names is not None:
            assert isinstance(name, str),\
                "arg 'name' should be a string"
            assert name not in self.colnames,\
                "arg 'name' should not be an existing column name"
        
        count_frame = self.count(column_names = column_names
                                 , name = name
                                 , wt = wt
                                 )
                                 
        if column_names is None:
            count_value = int(count_frame.to_pandas().iloc[0, 0])
            res = self.mutate({name: np.array(count_value)})
        else:
            res = self.left_join(count_frame, on = column_names)
        
        return res
    
    # pivot methods
    def pivot_wider(self
                    , names_from
                    , values_from
                    , values_fill = None
                    , values_fn = None
                    , id_cols = None
                    , sep = "__"
                    ):
        '''
        Pivot data from long to wide
        
        Parameters
        ----------
        names_from: string or list of strings
            column names whose unique combinations are expected to become column
            new names in the result
        
        values_from: string or list of strings
            column names to fill the new columns with
        
        values_fill: scalar (default is None)
            A value to fill the missing values with
            When None, missing values are left as-is
        
        values_fn: function or a dict of functions (default is None)
            A function to handle multiple values per row in the result.
            When a dict, keys should be same as arg 'values_from'
            When None, multiple values are in kept a list and a single value is
            not kept in a list (will be a scalar)
        
        id_cols: string or list of strings, default is None
            Names of the columns that should uniquely identify an observation 
            (row) after widening (columns of the original dataframe that are
            supposed to stay put)
            
        sep: string
            seperator to use while creating resulting column names
            
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> import numpy as np
        >>> 
        >>> # generic widening leads to list-columns
        >>> penguins_tidy.pivot_wider(id_cols       = "island"
        >>>                           , names_from  = "sex"
        >>>                           , values_from = "bill_length_mm"
        >>>                           )
        >>> 
        >>> # aggregate with a function
        >>> penguins_tidy.pivot_wider(id_cols       = "island"
        >>>                           , names_from  = "sex"
        >>>                           , values_from = "bill_length_mm"
        >>>                           , values_fn   = np.mean
        >>>                           )
        >>>                           
        >>> # choose different aggregation logic for value_from columns
        >>> penguins_tidy.pivot_wider(
        >>>   id_cols       = "island"
        >>>   , names_from  = "species"
        >>>   , values_from = ["bill_length_mm", "bill_depth_mm"]
        >>>   , values_fn   = {"bill_length_mm" : np.mean, "bill_depth_mm" : list}
        >>>   )
        >>>                           
        >>> # aggregate with almost any function
        >>> penguins_tidy.pivot_wider(
        >>>   id_cols       = "island"
        >>>   , names_from  = "species"
        >>>   , values_from = "sex"
        >>>   , values_fn   = lambda x: dict(pd.Series(x).value_counts())
        >>>   )
        >>> 
        >>> # All three inputs: 'id_cols', 'names_from', 'values_from' can be lists
        >>> penguins_tidy.pivot_wider(
        >>>     id_cols       = ["island", "sex"]
        >>>     , names_from  = "species"
        >>>     , values_from = "bill_length_mm"
        >>>     )
        >>>                           
        >>> penguins_tidy.pivot_wider(
        >>>     id_cols       = ["island", "sex"]
        >>>     , names_from  = "species"
        >>>     , values_from = ["bill_length_mm", "bill_depth_mm"]
        >>>     )
        >>> 
        >>> penguins_tidy.pivot_wider(id_cols       = ["island", "sex"]
        >>>                           , names_from  = ["species", "year"]
        >>>                           , values_from = "bill_length_mm"
        >>>                           )
        >>>                           
        >>> penguins_tidy.pivot_wider(
        >>>     id_cols       = ["island", "sex"]
        >>>     , names_from  = ["species", "year"]
        >>>     , values_from = ["bill_length_mm", "bill_depth_mm"]
        >>>     )
        >>> 
        >>> # when id_cols is empty, all columns except the columns from
        >>> # `names_from` and `values_from` are considered as id_cols
        >>> (penguins_tidy
        >>>  .select(['flipper_length_mm', 'body_mass_g'], include = False)
        >>>  .pivot_wider(names_from    = ["species", "year"]
        >>>               , values_from = ["bill_length_mm", "bill_depth_mm"]
        >>>               )
        >>>  )
        >>>                           
        >>> # fill the missing values with something
        >>> penguins_tidy.pivot_wider(id_cols       = "island"
        >>>                           , names_from  = "species"
        >>>                           , values_from = "bill_length_mm"
        >>>                           , values_fn   = np.mean
        >>>                           , values_fill = 0
        >>>                           )
        '''
        
        cn = self.colnames
        
        assert _is_string_or_string_list(names_from),\
            "arg 'names_from' should be string or a list of strings"
        names_from = _enlist(names_from)
        assert _is_unique_list(names_from),\
            "arg 'names_from' should have unique strings"
        assert set(names_from).issubset(cn),\
            "arg 'names_from' should be a subset of existing column names"
        
        assert _is_string_or_string_list(values_from),\
            "arg 'values_from' should be string or a list of strings"
        values_from = _enlist(values_from)
        assert set(values_from).issubset(cn),\
            "arg 'names_from' should have unique strings"
        assert len(set(values_from).intersection(names_from)) == 0,\
            ("arg 'names_from' and 'values_from' should not "
             "have common column names"
             )
        names_values_from = set(values_from).union(names_from)
        
        if id_cols is None:
            id_cols = list(set(cn).difference(names_values_from))
            if len(id_cols) == 0:
                raise Exception(
                    ("'id_cols' is turning out to be empty. Choose the "
                     "'names_from' and 'values_from' appropriately or specify "
                     "'id_cols' explicitly."
                     ))
            else:
                print("'id_cols' chosen: " + str(id_cols))
        else:
            assert _is_string_or_string_list(id_cols),\
                "arg 'id_cols' should be string or a list of strings"
            id_cols = _enlist(id_cols)
            assert _is_unique_list(id_cols),\
                "arg 'id_cols' should have unique strings"
            assert len(set(id_cols).intersection(names_values_from)) == 0,\
                ("arg 'id_cols' should not have common names with either "
                 "'names_from' or 'values_from'"
                 )
        if values_fill is not None:
            assert np.isscalar(values_fill),\
                "arg 'values_fill' should be a scalar"
        
        assert (values_fn is None 
                or callable(values_fn) 
                or isinstance(values_fn, dict)),\
            "When not None, arg 'values_fn' should be a callable or dict"
        
        if values_fn is None:
            
            def unlist_scalar(x):
                if isinstance(x, list) and len(x) == 1:
                    x = x[0]
                return x
            
            values_fn = unlist_scalar
            
        elif isinstance(values_fn, dict):
            assert isinstance(values_fn, dict),\
                ("When arg 'values_fn' is not a scalar, it should either be "
                 "a dict"
                 )
            keys_fn = set(values_fn.keys())
            assert set(values_from) == keys_fn,\
                ("When arg 'values_fn' is a dict, keys should match "
                 "arg 'values_from'"
                 )
            assert all([callable(x) for x in values_fn.values()]),\
                "When arg 'values_fn' is a dict, values should be functions"
        else:
            pass
        
        assert isinstance(sep, str),\
            "arg 'sep' should be a string"
        
        # make values_from a scalar if it is
        if len(values_from) == 1:
            values_from = values_from[0]

        res = pd.pivot_table(data         = self.__data
                             , index      = id_cols
                             , columns    = names_from
                             , values     = values_from
                             , fill_value = values_fill
                             , aggfunc    = values_fn
                             , margins    = False
                             , dropna     = False
                             , observed   = True
                             )
        return tidyframe(res)
    
    def pivot_longer(self
                     , cols
                     , names_to = "name"
                     , values_to = "value"
                     , include = True
                     , values_drop_na = False
                     ):
        '''
        Pivot from wide to long
        aka melt
        
        Parameters
        ----------
        cols: list of strings
          Column names to be melted.
          The dtypes of the columns should match.
          Leftover columns are considered as 'id' columns.
          When include is False, 'cols' refers to leftover columns.
        names_to: string (default: 'name')
          Name of the resulting column which will hold the names of the columns
          to be melted
        values_to: string (default: 'value')
          Name of the resulting column which will hold the values of the columns
          to be melted
        include: bool
          If True, cols are used to melt. Else, cols are considered as 'id'
          columns
        values_drop_na: bool
          Whether to drop the rows corresponding to missing value in the result
          
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # pivot to bring values from columns ending with 'mm'
        >>> cns = ['species'
        >>>        , 'bill_length_mm'
        >>>        , 'bill_depth_mm'
        >>>        , 'flipper_length_mm'
        >>>        ]
        >>> (penguins_tidy.select(cns)
        >>>               .pivot_longer(cols = ['bill_length_mm',
        >>>                                     'bill_depth_mm',
        >>>                                     'flipper_length_mm']
        >>>                             )
        >>>               )
        >>>               
        >>> # pivot by specifying 'id' columns to obtain the same result as above
        >>> # this is helpful when there are many columns to melt
        >>> (penguins_tidy.select(cns)
        >>>               .pivot_longer(cols = 'species',
        >>>                             include = False
        >>>                             )
        >>>               )
        '''
        # assertions
        cn = self.colnames
        assert _is_string_or_string_list(cols),\
            "arg 'cols' should be a string or a list of strings"
        cols = _enlist(cols)
        assert set(cols).issubset(cn),\
            "arg 'cols' should be a subset of existing column names"
        assert isinstance(include, bool),\
            "arg 'include' should be a bool"
        if not include:
           cols = list(setlist(cn).difference(cols))
           assert len(cols) > 0,\
               "At least one column should be selected for melt"
        
        id_vars = set(cn).difference(cols)
        assert isinstance(names_to, str),\
            "arg 'names_to' should be a string"
        assert isinstance(values_to, str),\
            "arg 'values_to' should be a string"
        assert names_to not in id_vars,\
            "arg 'names_to' should not match a id column"
        assert values_to not in id_vars,\
            "arg 'values_to' should not match a id column"
        
        # core operation
        res = (self.__data
                   .melt(id_vars         = id_vars
                         , value_vars    = cols
                         , var_name      = names_to
                         , value_name    = values_to
                         , ignore_index  = True
                         )
                   )
                   
        res = tidyframe(res)
        
        if values_drop_na:
            res = res.drop_na(values_to)
        
        return res
    
    ##########################################################################
    # slice extensions
    ##########################################################################
    
    def slice_head(self
                   , n = None
                   , prop = None
                   , rounding_type = "round"
                   , by = None
                   ):
        '''
        Subset first few rows
        
        Parameters
        ----------
        n: int
            Number of rows to subset, should be atleast 1
        prop: float
            proportion of rows to subset, should be between 0(exclusive) and 1 
            (inclusive)
        rounding_type: string
            When prop is provided, rounding method to be used.
            Options: 'round' (default), 'floor', 'ceiling'
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.
        2. Row order is preserved by the method.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy.slice_head(n = 3)
        >>> penguins_tidy.slice_head(prop = 0.01)
        >>> 
        >>> penguins_tidy.slice_head(n = 1, by = 'species')
        >>> penguins_tidy.slice_head(prop = 0.01, by = 'species')
        '''
        nr = self.nrow
        cn = self.colnames

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
            
        if by is None:
            assert n <= self.__data.shape[0],\
                "arg 'n' should not exceed the number of rows of the dataframe"
            res = tidyframe(self.__data.head(n)
                                      , check = False
                                      , copy = False
                                      )
        else:
            self._validate_by(by)
            by = _enlist(by)
            if not case_prop:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                
                ro_name = _generate_new_string(cn) 
                res = (self.group_modify(lambda x: x.slice(np.arange(n))
                                         , by = by
                                         , preserve_row_order = True
                                         , row_order_column_name = ro_name
                                         )
                           )
            else:
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    ("arg 'rounding_type' should be one among: 'round' (default)"
                     " , 'ceiling', 'floor'"
                     )
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                
                ro_name = _generate_new_string(cn)    
                res = self.group_modify(
                          lambda x: x.slice(range(int(roundf(x.shape[0] * prop))))
                          , by = by
                          , preserve_row_order = True
                          , row_order_column_name = ro_name
                          )
            
        return res.relocate(cn)
    
    head = slice_head
    
    def slice_tail(self
                   , n = None
                   , prop = None
                   , rounding_type = "round"
                   , by = None
                   ):
        '''
        Subset last few rows
        
        Parameters
        ----------
        n: int
            Number of rows to subset, should be atleast 1
        prop: float
            proportion of rows to subset, should be between 0(exclusive) and 1(inclusive)
        rounding_type: string
            When prop is provided, rounding method to be used. 
            Options: 'round' (default), 'floor', 'ceiling'
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.
        2. Row order is preserved by the method.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy.slice_tail(n = 3)
        >>> penguins_tidy.slice_tail(prop = 0.01)
        >>> 
        >>> penguins_tidy.slice_tail(n = 1, by = 'species')
        >>> penguins_tidy.slice_tail(prop = 0.01, by = 'species')
        '''
        nr = self.nrow
        cn = self.colnames

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
            
        if by is None:
            assert n <= self.__data.shape[0],\
                "arg 'n' should not exceed the number of rows of the dataframe"
            res = tidyframe(self.__data.tail(n).reset_index(drop = True)
                                      , check = False
                                      )
        else:
            self._validate_by(by)
            by = _enlist(by)
            if not case_prop:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                
                ro_name = _generate_new_string(cn) 
                res = (self.group_modify(lambda x: x.tail(n).reset_index(drop = True)
                                             , by = by
                                             , is_pandas_udf = True
                                             , preserve_row_order = True
                                             , row_order_column_name = ro_name
                                             )
                           )
            else:
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    "arg 'ties_method' should be one among: 'round' (default), 'ceiling', 'floor'"
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor
                
                ro_name = _generate_new_string(cn)    
                res = (self.group_modify(
                    lambda x: (x.tail(int(roundf(x.shape[0] * prop)))
                                .reset_index(drop = True)
                                )
                    , by = by
                    , is_pandas_udf      = True
                    , preserve_row_order = True
                    , row_order_column_name = ro_name
                    ))
            
        return res.relocate(cn)
    
    tail = slice_tail
    
    def slice_sample(self
                     , n            = None
                     , prop         = None
                     , replace      = False
                     , weights      = None
                     , random_state = None
                     , by           = None
                     ):
        '''
        Sample a few rows
        
        Parameters
        ----------
        n: int
            Number of rows to sample, should be atleast 1
        prop: float
            Proportion of rows to subset, should be non-negative
        replace: bool, default is False
            Whether to sampling should be done with replacement
        weights: string, pandas series, numpy array (Default is None)
            When a string, it should be an column of numeric dtype
            When None, no weights are used
        random_state: positive integer (Default is None)
            Seed to keep the sampling reproducible
            When None, a pseudorandom seed is chosen
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.
        2. Row order is not preserved by the method.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # swor: sample without replacement
        >>> # swr: sample with replacement
        >>> 
        >>> # sample by specifiying count
        >>> penguins_tidy.slice_sample(n = 5)                    # swor
        >>> penguins_tidy.slice_sample(n = 5, replace = True)    # swr, smaller than input
        >>> penguins_tidy.slice_sample(n = 1000, replace = True) # swr, larger than input
        >>> 
        >>> # sample by specifiying proportion of number of rows of the input
        >>> penguins_tidy.slice_sample(prop = 0.3)                 # swor
        >>> penguins_tidy.slice_sample(prop = 0.3, replace = True) # swr, smaller than input
        >>> penguins_tidy.slice_sample(prop = 1.1, replace = True) # swr, larger than input
        >>> 
        >>> # sample with weights
        >>> penguins_tidy.slice_sample(prop = 0.3, weights = 'year')
        >>> 
        >>> # sampling is reproducible by setting a random state
        >>> penguins_tidy.slice_sample(n = 3, random_state = 42)
        >>> 
        >>> # per group sampling
        >>> penguins_tidy.slice_sample(n = 5, by = 'species', random_state = 1)
        '''
        nr = self.nrow
        cn = self.colnames

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int"
            assert prop > 0,\
                "arg 'prop' should be a positive float or int"
        
        assert isinstance(replace, bool),\
            "Arg 'replace' should be True or False"
        if prop is not None:
            if prop > 1 and replace is False:
                raise Exception("When arg 'prop' > 1, arg 'replace' should be True")
            
        if n is not None:
            if n > nr and replace is False:
                raise Exception("When arg 'n' is greater than the number of rows of the input, arg 'replace' should be True")
            
        # Arg weights is not validated
        # Arg random_state is not validated
        
        if by is None:
            if n is not None:
                res = (self.__data
                           .sample(n              = n
                                   , axis         = "index"
                                   , replace      = replace
                                   , weights      = weights
                                   , random_state = random_state
                                   )
                           .reset_index(drop = True)
                           )
            else:
                res = (self.__data
                           .sample(frac           = prop
                                   , axis         = "index"
                                   , replace      = replace
                                   , weights      = weights
                                   , random_state = random_state
                                   )
                           .reset_index(drop = True)
                           )
        else:
            by = _enlist(by)
            self._validate_by(by)
            n_groups = (self.__data
                            .groupby(by, sort = False, dropna = False)
                            .ngroups
                            )
            
            if random_state is not None:
                np.random.seed(random_state)
            seeds = np.random.choice(n_groups * 100, n_groups, replace = False)
            
            groups_frame = self.__data.loc[:, by].drop_duplicates()
            grp_name = _generate_new_string(self.colnames)
            groups_frame[grp_name] = np.arange(groups_frame.shape[0]) 
            groups_frame["__seed"] = seeds
            
            if n is not None:
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    "arg 'n' should not exceed the size of any chunk after grouping"
                
                def sample_chunk_n(x):
                    rs = x['__seed'].iloc[0]
                    res = (x.sample(n              = n
                                    , axis         = "index"
                                    , replace      = replace
                                    , weights      = weights
                                    , random_state = rs
                                    )
                            .reset_index(drop = True)
                            )
                    return res
                
                res = (self.__data
                           .merge(groups_frame, on = by, how = 'left')
                           .groupby(grp_name, sort = False, dropna = False)
                           .apply(sample_chunk_n)
                           .reset_index(drop = True)
                           .drop(columns = ["__seed", grp_name])
                           )
            else:
                def sample_chunk_prop(x):
                    rs = x['__seed'].iloc[0]
                    res = (x.sample(frac           = prop
                                    , axis         = "index"
                                    , replace      = replace
                                    , weights      = weights
                                    , random_state = rs
                                    )
                            .reset_index(drop = True)
                            )
                    return res
                
                res = (self.__data
                           .merge(groups_frame, on = by, how = 'left')
                           .groupby(grp_name, sort = False, dropna = False)
                           .apply(sample_chunk_prop)
                           .reset_index(drop = True)
                           .drop(columns = ["__seed", grp_name])
                           )
            
            res = (res.sample(frac = 1, random_state = random_state)
                      .reset_index(drop = True)
                      )
            
        return tidyframe(res).relocate(cn)
    
    sample = slice_sample
    
    ##########################################################################
    # slice min/max
    ##########################################################################
    
    def slice_min(self
                  , n = None
                  , prop = None
                  , order_by_column = None
                  , with_ties = True
                  , rounding_type = "round"
                  , by = None
                  ):
        '''
        Subset top rows ordered by some columns
        
        Parameters
        ----------
        n: int
            Number of rows to subset, should be atleast 1
        prop: float
            Proportion of rows to subset, should be non-negative
        order_by_column: string or list of strings
            column to order by
        with_ties: bool (default: True)
            Whether to return all rows when ordering results in ties. If True, 
            the output might have more than n/prop rows
        rounding_type: string (Default is None)
            Rounding type to used when prop is provided. 'rounding_type' should 
            be one among: 'round', 'ceiling', 'floor'
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.S
        2. Row order is not preserved by the method.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # subset atleast 4 rows corresponding to ascending 'body_mass_g'
        >>> penguins_tidy.slice_min(n = 4, order_by_column = 'body_mass_g')
        >>> 
        >>> # subset exactly 4 rows corresponding to ascending 'body_mass_g'
        >>> # when more than 'n' rows exist, top n rows in the row order are chosen
        >>> penguins_tidy.slice_min(n = 4, order_by_column = 'body_mass_g', with_ties = False)
        >>> 
        >>> # subset a fraction of rows corresponding to ascending 'body_mass_g'
        >>> penguins_tidy.slice_min(prop = 0.1, order_by_column = 'body_mass_g')
        >>> 
        >>> # subset atleast 3 rows corresponding to ascending 'bill_length_mm' per each 'species'
        >>> penguins_tidy.slice_min(n = 2, order_by_column = 'bill_length_mm', by = 'sex')
        >>> 
        >>> # subset atleast 1% corresponding to ascending 'bill_length_mm' per each 'species'
        >>> penguins_tidy.slice_min(prop = 0.01, order_by_column = 'bill_length_mm', by = 'sex')
        >>> 
        >>> # order by column 'bill_length_mm' and break ties using 'bill_depth_mm'
        >>> # in decreasing order and then pick the min 5 rows
        >>> (penguins_tidy
        >>>  .mutate({'rank': (lambda x, y: dense_rank([x, y],
        >>>                                            ascending = [True, False]
        >>>                                            ),
        >>>                    ["bill_length_mm", "bill_depth_mm"]
        >>>                    )
        >>>          })
        >>>  .slice_min(n = 5, order_by_column = "rank")
        >>>  .select("rank", include = False)
        >>>  )
        '''
        nr = self.nrow
        cn = self.colnames

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
            
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
        
        if order_by_column is None:
            raise Exception("arg 'order_by' should not be None")
        else:
            assert order_by_column in self.colnames,\
                "'order_by_column' should be a existing column name"
          
        assert isinstance(with_ties, bool),\
            "arg 'with_ties' should be a bool"
            
        if with_ties:
            keep_value = "all"
        else:
            keep_value = "first"
        
        if by is None:
            ro_name = _generate_new_string(self.colnames)
            res = (self.add_row_number(name = ro_name)
                       .to_pandas(copy = False)
                       .nsmallest(n, columns = order_by_column, keep = keep_value)
                       .reset_index(drop = True)
                       .sort_values(ro_name, ignore_index = True)
                       .loc[:, cn]
                       .pipe(lambda x: tidyframe(x, check = False))
                       )
        else:
            self._validate_by(by)
            by = _enlist(by)
            
            if case_prop: # grouped prop
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    ("arg 'ties_method' should be one among: 'round' (default),"
                     " 'ceiling', 'floor'"
                     )
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor

                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nsmallest(int(roundf(x.shape[0] * prop))
                                                    , columns = order_by_column
                                                    , keep = keep_value
                                                    )
                                          .pipe(tidyframe
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              , row_order_column_name = _generate_new_string(cn)
                              )
                           .select(cn)
                           .arrange(order_by_column, by = by)
                           )
            else: # grouped n
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    ("arg 'n' should not exceed the size of any chunk after "
                    "grouping")
                
                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nsmallest(n
                                                    , columns = order_by_column
                                                    , keep = keep_value
                                                    )
                                          .pipe(tidyframe
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              , row_order_column_name = _generate_new_string(cn)
                              )
                           .select(cn)
                           .arrange(order_by_column, by = by)
                           )
        return res.relocate(cn)
    
    def slice_max(self
                  , n = None
                  , prop = None
                  , order_by_column = None
                  , with_ties = True
                  , rounding_type = "round"
                  , by = None
                  ):
        '''
        Subset top rows ordered by some columns
        
        Parameters
        ----------
        n: int
            Number of rows to subset, should be atleast 1
        prop: float
            Proportion of rows to subset, should be non-negative
        order_by_column: string or list of strings
            column to order by
        with_ties: bool (default: True)
            Whether to return all rows when ordering results in ties. If True, 
            the output might have more than n/prop rows
        rounding_type: string (Default is None)
            Rounding type to used when prop is provided. 'rounding_type' should 
            be one among: 'round', 'ceiling', 'floor'
        by: string or list of strings
            column names to group by
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. Only one argument among 'n' and 'prop' should be provided.
        2. Row order is not preserved by the method.
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> # subset atleast 4 rows corresponding to descending 'body_mass_g'
        >>> penguins_tidy.slice_max(n = 4, order_by_column = 'body_mass_g')
        >>> 
        >>> # subset exactly 4 rows corresponding to descending 'body_mass_g'
        >>> # when more than 'n' rows exist, top n rows in the row order are chosen
        >>> penguins_tidy.slice_max(n = 4, order_by_column = 'body_mass_g', with_ties = False)
        >>> 
        >>> # subset a fraction of rows corresponding to descending 'body_mass_g'
        >>> penguins_tidy.slice_max(prop = 0.1, order_by_column = 'body_mass_g')
        >>> 
        >>> # subset atleast 3 rows corresponding to descending 'bill_length_mm' per each 'species'
        >>> penguins_tidy.slice_max(n = 2, order_by_column = 'bill_length_mm', by = 'sex')
        >>> 
        >>> # subset atleast 1% corresponding to descending 'bill_length_mm' per each 'species'
        >>> penguins_tidy.slice_max(prop = 0.01, order_by_column = 'bill_length_mm', by = 'sex')
        >>> 
        >>> # order by column 'bill_length_mm' and break ties using 'bill_depth_mm'
        >>> # in decreasing order and then pick the min 5 rows
        >>> (penguins_tidy
        >>>  .mutate({'rank': (lambda x, y: dense_rank([x, y],
        >>>                                            ascending = [True, False]
        >>>                                            ),
        >>>                    ["bill_length_mm", "bill_depth_mm"]
        >>>                    )
        >>>          })
        >>>  .slice_max(n = 5, order_by_column = "rank")
        >>>  .select("rank", include = False)
        >>>  )
        '''
        nr = self.nrow
        cn = self.colnames

        # exactly one of then should be none
        assert not ((n is None) and (prop is None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
        assert not ((n is not None) and (prop is not None)),\
            "Exactly one arg among 'n', 'prop' should be provided"
            
        if n is not None:
            assert isinstance(n, int),\
                "arg 'n' should be a positive integer"
            assert n > 0,\
                "arg 'n' should be atleast 1"
            case_prop = False
            
        if prop is not None:
            assert isinstance(prop, (float, int)),\
                "arg 'prop' should be a positive float or int not exceeding 1"
            assert prop > 0 and prop <= 1,\
                "arg 'prop' should be a positive float or int not exceeding 1"
            n = int(np.round(prop * nr))
            case_prop = True
        
        if order_by_column is None:
            raise Exception("arg 'order_by' should not be None")
        else:
            assert order_by_column in self.colnames,\
                "'order_by_column' should be a existing column name"
          
        assert isinstance(with_ties, bool),\
            "arg 'with_ties' should be a bool"
            
        if with_ties:
            keep_value = "all"
        else:
            keep_value = "first"
        
        if by is None:
            ro_name = _generate_new_string(self.colnames)
            res = (self.add_row_number(name = ro_name)
                       .to_pandas(copy = False)
                       .nlargest(n, columns = order_by_column, keep = keep_value)
                       .reset_index(drop = True)
                       .sort_values(ro_name, ignore_index = True)
                       .loc[:, cn]
                       .pipe(lambda x: tidyframe(x, check = False))
                       )
        else:
            self._validate_by(by)
            by = _enlist(by)
            
            if case_prop:
                assert isinstance(rounding_type, str),\
                    "arg 'ties_method' should be a string"
                assert rounding_type in ['round', 'ceiling', 'floor'],\
                    ("arg 'ties_method' should be one among: 'round' (default),"
                     " 'ceiling', 'floor'"
                     )
                
                if rounding_type == "round":
                    roundf = np.round
                elif rounding_type == "ceiling":
                    roundf = np.ceil
                else:
                    roundf = np.floor

                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nlargest(int(roundf(x.shape[0] * prop))
                                                    , columns = order_by_column
                                                    , keep = keep_value
                                                    )
                                          .pipe(tidyframe
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              , row_order_column_name = _generate_new_string(cn)
                              )
                           .select(cn)
                           .arrange(order_by_column, by = by)
                      )
            else: # grouped n
                min_group_size = (self.__data
                                      .groupby(by, sort = False, dropna = False)
                                      .size()
                                      .min()
                                      )
                if n > min_group_size:
                    print("Minimum group size is ", min_group_size)
                assert n <= min_group_size,\
                    ("arg 'n' should not exceed the size of any chunk after "
                    "grouping")
                
                res = (self.group_modify(
                              lambda x: (x.to_pandas(copy = False)
                                          .nlargest(n
                                                    , columns = order_by_column
                                                    , keep = keep_value
                                                    )
                                          .pipe(tidyframe
                                                , copy = False
                                                , check = False
                                                )
                                        )
                              , by = by
                              , preserve_row_order = True
                              , row_order_column_name = _generate_new_string(cn)
                              )
                           .select(cn)
                           .arrange(order_by_column, by = by)
                           )
        return res.relocate(cn)
    
    # set like methods
    def union(self, y):
        '''
        Union of rows y with the self
        Equivalent to outer join over all columns
        
        Parameters
        ----------
        y: tidyframe with same columns as x
        
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy[0:5, :].union(penguins_tidy[1:6, :])
        '''
        assert set(self.colnames) == set(y.colnames),\
            "union expects the column names to match"
        res = (pd.concat([self.__data, y.to_pandas(copy = False)])
                 .drop_duplicates(keep = "first")
                 .reset_index(drop = True)
                 )
        return tidyframe(res, check = False, copy = False)
    
    def intersection(self, y):
        '''
        Intersection of rows y with the self
        Equivalent to inner join over all columns
        
        Parameters
        ----------
        y: tidyframe with same columns as x
        
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy[0:5, :].intersection(penguins_tidy[1:6, :])
        '''
        assert set(self.colnames) == set(y.colnames),\
            "intersection expects the column names to match"

        res = (pd.concat([self.__data, y.to_pandas(copy = False)])
                 .assign(**{"__is_dup": lambda x: x.duplicated(keep = False)})
                 .iloc[np.arange(self.nrow), :]
                 .query("__is_dup")
                 .drop(columns = "__is_dup")
                 .reset_index(drop = True)
                 )
        return tidyframe(res, check = False, copy = False)
      
    def setdiff(self, y):
        '''
        Keep rows of self which are not in y
        Equivalent to anti join over all columns
        
        Parameters
        ----------
        y: tidyframe with same columns as x
        
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy[0:5, :].setdiff(penguins_tidy[1:6, :])
        '''
        assert set(self.colnames) == set(y.colnames),\
            "setdiff expects the column names to match"

        res = (pd.concat([self.__data, y.to_pandas(copy = False)])
                 .assign(**{"__is_dup": lambda x: x.duplicated(keep = False)})
                 .iloc[np.arange(self.nrow), :]
                 .query("~__is_dup")
                 .drop(columns = "__is_dup")
                 .reset_index(drop = True)
                 )
        
        return tidyframe(res, check = False, copy = False)
    # na handling methods
    
    ############################################################################
    # any_na
    ############################################################################
    def any_na(self):
        '''
        Is there a missing value in the dataframe?
        
        Returns
        -------
        bool
            True if there is a missing value, False otherwise
        '''
        res = (self.__data
                   .isna() # same dim as input
                   .any()  # series of booleans per column
                   .any()  # single value
                   )
                   
        return bool(res)
    
  ##############################################################################
  # replace_na
  ##############################################################################
    
    def replace_na(self, value):
        '''
        Replace missing values with a specified value
        
        Parameters
        ----------
        value: dict or a scalar
            When a dict, key should be a column name and value should be the
            value to replace by missing values of the column
            When a scalar, missing values of all columns will be replaced with
            value
        
        Returns
        -------
        tidyframe
            
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy.replace_na({'sex': 'unknown'})
        >>> penguins_tidy.select(predicate = dtypes.is_numeric_dtype).replace_na(1)
        '''
        return tidyframe(self.__data.fillna(value).fillna(pd.NA)
                          , copy = False
                          , check = False
                          )
    
    def drop_na(self, column_names = None):
        '''
        Drops rows if missing values are present in specified columns
        
        Parameters
        ----------
        column_names: string or list of strings
            Drop a row if there is a missing value in any of the 
            column_names. When None, all columns are considered.
            
        Returns
        -------
        tidyframe
            
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy.drop_na() # remove a row if any column has a NA
        >>> # remove a row only if there is a missing value in 'bill_length_mm'
        >>> # column
        >>> penguins_tidy.drop_na('bill_length_mm')
        '''
        cn = self.colnames
        if column_names is not None:
            self._validate_column_names(column_names)
            column_names = _enlist(column_names)
        else:
            column_names = cn
        
        res = (self.__data
                   .dropna(axis = "index"
                           , how = "any"
                           , subset = column_names
                           , inplace = False
                           )
                   .reset_index(drop = True)
                   )
        
        return tidyframe(res, check = False, copy = False)
    
    ############################################################################
    # fill_na (fill)
    ############################################################################
        
    def fill_na(self
                , column_direction_dict
                , order_by = None
                , ascending = True
                , na_position = "last"
                , by = None
                ):
        '''
        Fill missing values from neighboring values per column
        
        Paramaters
        ----------
        column_direction_dict: dict
            where key is a column name and value is the direction to fill.
            Direction should be one among: 'up', 'down', 'updown' and 'downup'
        order_by: string or list of strings
            Column names to order by before filling
        ascending: bool or a list of bools (default: True)
            Used when order_by is specified
        na_position: string (default: 'last')
            One among: 'first', 'last'
        by: string or list of strings
            Column names to group by 
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. The output might still have missing values after applying fill_na.
        2. The method preserves the row order.
        
        Examples
        --------
        >>> df = tidyframe(
        >>>          pd.DataFrame({'A': [pd.NA, 1, 1, 2, 2, 3, pd.NA],
        >>>                        'B': [pd.NA, pd.NA, 1, 2, pd.NA, pd.NA, 3],
        >>>                        'C': [1, 2, 1, 2, 1, 2, 1]
        >>>                        }
        >>>                       )
        >>>          )
        >>> df
        >>> 
        >>> df.fill_na({'B': 'up'})
        >>> df.fill_na({'B': 'down'})
        >>> df.fill_na({'B': 'downup'})
        >>> df.fill_na({'B': 'updown'})
        >>> 
        >>> df.fill_na({'B': 'up'}, by = 'A')
        >>> df.fill_na({'B': 'down'}, by = 'A')
        >>> df.fill_na({'B': 'updown'}, by = 'A')
        >>> df.fill_na({'B': 'downup'}, by = 'A')
        >>> 
        >>> df.fill_na({'B': 'updown'}, order_by = "C")
        >>> df.fill_na({'B': 'updown'}, order_by = "C", by = "A")
        '''
        cn = self.colnames
        assert isinstance(column_direction_dict, dict),\
            "arg 'column_direction_dict' should be a dict"
        assert set(column_direction_dict.keys()).issubset(cn),\
            ("keys of 'column_direction_dict' should be a subset of existing "
             "column names"
             )
        valid_methods = ["up", "down", "updown", "downup"]
        assert set(column_direction_dict.values()).issubset(valid_methods),\
            ("values of 'column_direction_dict' should be one among:  "
             "'up', 'down', 'updown', 'downup'"
             )
        
        if order_by is not None:
            order_by = self._clean_order_by(order_by)
        
        if by is not None:
            self._validate_by(by)
            by = _enlist(by)
            
            # by columns should not be altered
            keys = column_direction_dict.keys()
            assert len(set(by).intersection(keys)) == 0,\
                ("'by' columns cannot be altered. keys of 'column_direction_dict'"
                 " should not intersect with 'by'"
                 )
            
            # by columns should not intersect with order_by columns
            if order_by is not None:
                assert len(set(by).intersection([x[0] for x in order_by])) == 0,\
                    "'by' columns should not intersect with 'order_by' columns"

        def fill_chunk(x, cdd):
            
            chunk = x.copy()
            
            for akey in cdd:
                method = cdd[akey]
                if method == "up":
                    chunk[akey] = chunk[akey].bfill()
                elif method == "down":
                    chunk[akey] = chunk[akey].ffill()
                elif method == "updown":
                    chunk[akey] = chunk[akey].bfill()
                    chunk[akey] = chunk[akey].ffill()
                else:
                    chunk[akey] = chunk[akey].ffill()
                    chunk[akey] = chunk[akey].bfill()
                
            return chunk
        
        if order_by is not None:
            ro_name = _generate_new_string(self.colnames)
            res = self.add_row_number(name = ro_name).arrange(order_by)
        else:
            res = self
        
        
        
        if by is not None:
            res = res.group_modify(
              lambda x: fill_chunk(x, column_direction_dict)
              , by = by
              , preserve_row_order = True
              , is_pandas_udf = True
              )
        else:
            res = (res.to_pandas(copy = False)
                      .pipe(lambda x: fill_chunk(x, column_direction_dict))
                      .pipe(lambda x: tidyframe(x, copy = False, check = False))
                      )
        
        if order_by is not None:
            res = res.arrange(ro_name).select(ro_name, include = False)
         
        return res
    
    fill = fill_na
    
    # string utilities
    def separate(self
                 , column_name
                 , into
                 , sep = '[^[:alnum:]]+'
                 , strict = True
                 , keep = False
                 ):
        
        '''
        Split a string column into multiple string columns rowwise
        
        Parameters
        ----------
        column_name: string
            Name of the column to split
        into: list of strings
            Name of the new columns after slitting the column
        sep: string
            Regex to identify the separator
        strict: bool (default: True)
            When True, results in an exception if column splits to exactly
            to result into 'into' number of columns.
        keep: bool (default: False)
            Whether to keep the original column after splitting
            
        Examples
        --------
        >>> df = tidyframe(pd.DataFrame({'col': ["a_b", "c_d", "e_f_g"]}))
        >>> print(df)
        >>> 
        >>> # separate into three columns
        >>> df.separate('col', into = ["col_1", "col_2", "col_3"], sep = "_")
        >>> 
        >>> # separate into two columns and ignore the last piece
        >>> df.separate('col', into = ["col_1", "col_2"], sep = "_", strict = False)
        '''
        
        assert isinstance(column_name, str),\
            "arg 'column_name' should be a string"
        assert column_name in self.colnames,\
            "arg 'column_name' should be an existing column name"
        assert _is_string_or_string_list(into),\
            "arg 'into' should be a list of strings"
        assert len(into) > 1,\
            "arg 'into' should be a list of strings"
        assert isinstance(sep, str),\
            "arg 'sep' should be a string"
        assert isinstance(strict, bool),\
            "arg 'strict' should be a bool"
        assert isinstance(keep, bool),\
            "arg 'keep' should be a bool" 
            
        # split  and form a pandas df
        split_df = (pd.DataFrame([re.split(sep, i) for i in self.pull(column_name)])
                      .fillna(pd.NA)
                      )
        
        if len(into) == split_df.shape[1]:
            split_df.columns = into
        elif len(into) < split_df.shape[1]: # more pieces than expected
            if strict:
                raise Exception(("Column is split into more number of columns"
                                 " than the length of 'into'"))
            else:
                # keep only 'into' columns
                split_df = split_df.iloc[:, 0:len(into)]
                split_df = split_df.set_axis(into, axis = 'columns')
        else: # less pieces than expected
            if strict:
                raise Exception(("Column is split into less number of columns"
                                 " than the length of 'into'"))
            else:
                split_df.columns = into[0:split_df.shape[1]]     
            
        if keep:
            res = self.cbind(tidyframe(split_df, check = False))
        else:
            res = (self.cbind(tidyframe(split_df, check = False, copy = False))
                       .select(column_name, include = False)
                       )
        return res
    
    def unite(self, column_names, into, sep = "_", keep = False):
        '''
        Split a string column into multiple string columns rowwise
        
        Parameters
        ----------
        column_name: string
            Name of the column to split
        into: list of strings
            Name of the new columns after slitting the column
        sep: string
            Regex to identify the separator
        strict: bool (default: True)
            When True, results in an exception if column splits to exactly
            to result into 'into' number of columns.
        keep: bool (default: False)
            Whether to keep the original column after splitting
            
        Examples
        --------
        >>> df = tidyframe(pd.DataFrame({'col': ["a_b", "c_d", "e_f_g"]}))
        >>> print(df)
        >>> 
        >>> # separate into three columns
        >>> (df.separate('col', into = ["col_1", "col_2", "col_3"], sep = "_")
        >>>    .unite(column_names = ["col_1", "col_2", "col_3"], into = "united", sep = "_")
        >>>    )
        >>> 
        >>> # separate into two columns and ignore the last piece
        >>> df.separate('col', into = ["col_1", "col_2"], sep = "_", strict = False)
        '''
        
        
        def reduce_join(df, columns, sep):
            assert len(columns) > 1
            slist = [df[x].astype(str) for x in columns]
            red_series = functools.reduce(lambda x, y: x + sep + y, slist[1:], slist[0])
            return red_series.to_frame(name = into)
                
        joined = reduce_join(self.__data, column_names, sep)
        
        if not keep:
           res = (self.cbind(tidyframe(joined, check = False))
                      .select(column_names, include = False)    
                      )
        else:
           res = self.cbind(tidyframe(joined, check = False)) 
         
        return res
    
    def separate_rows(self, column_name, sep = ';'):
        '''
        split a string column using a seperator and create a row for each
        
        Parameters
        ----------
        column_name: string
            A column name to split
        
        sep: string
            regex to split
            
        Returns
        -------
        tidyframe
        
        Examples
        --------
        >>> temp = tidyframe(pd.DataFrame({"A": ["hello;world", "hey,mister;o/mister"]}))
        >>> temp
        >>> temp.separate_rows('A', sep = ",|;")
        '''
        
        assert isinstance(column_name, str),\
            "arg 'column_name' should be a string"
        assert column_name in self.colnames,\
            "arg 'column_name' should be an exisiting column name"
        assert isinstance(sep, str),\
            "arg 'sep' should be a string"
        
        def splitter(str_col):
            return [re.split(sep, x) for x in str_col]
            
        res = (self.__data
                   .assign(**{column_name: lambda x: splitter(x[column_name])})
                   .explode(column_name, ignore_index = True)
                   )
        
        return tidyframe(res, check = False, copy = False)
    
    ##########################################################################
    # nest and unnest
    ##########################################################################
    
    
    ##########################################################################
    # nest_by
    ##########################################################################
    def nest_by(self
                , by = None
                , nest_column_name = 'data'
                , drop_by = True
                ):
        '''
        Nest all columns of tidy dataframe with respect to 'by' columns
        
        Parameters
        ----------
        by: str or list of strings
            Columns to stay, rest of them are nested
        nest_column_name: str
            Name of the resulting nested column (pandas Series)
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. 'by' should not contain all column names (some columns should be left
           for nesting)
           
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())

        >>> penguins_tidy.nest_by(by = 'species')
        >>> penguins_tidy.nest_by(by = ['species', 'sex'])
        '''
        cn = self.colnames
        self._validate_by(by)
        by = _enlist(by)
        assert len(by) < len(cn),\
            "arg 'by' should not contain all exisiting column names"
        
        assert nest_column_name not in cn,\
            "arg 'nest_column_name' should not be an exisiting column name"
            
        assert isinstance(drop_by, bool),\
            "arg 'drop_by' should be a bool"
        
        pdf = self.to_pandas(copy = False)
        
        extra = False
        if len(by) == 1:
            extra = True
            new_colname = _generate_new_string(cn)
            by = [new_colname] + by # note that by has changed
    
        if extra:
            go = (pdf.assign(**{new_colname: 1})
                     .groupby(by, sort = False, dropna = False)
                     )
        else:
            go = pdf.groupby(by, sort = False, dropna = False)
                      
        res = pd.DataFrame(list(go.groups.keys()), columns = by)
          
        
        # add data into nest_column_name column
        if drop_by:
            res[nest_column_name] = (
                pd.Series(map(lambda x: x.drop(columns = by)
                              , dict(tuple(go)).values()
                              )
                          )
                )
        else:
            # drop the extra column in the nested column
            if extra:
                res[nest_column_name] = (
                    pd.Series(map(lambda x: x.drop(columns = new_colname)
                                  , dict(tuple(go)).values()
                                  )
                              )
                    )
            else:
                res[nest_column_name] = pd.Series(dict(tuple(go)).values())
        
        if extra:
            res = res.drop(columns = new_colname)
        
        res = res.convert_dtypes().fillna(pd.NA)
        
        res[nest_column_name] = ( 
            res[nest_column_name].apply(lambda x: tidyframe(
                                                    x
                                                    , copy = False
                                                    , check = False
                                                    )
                                        )
                                )
        
        return tidyframe(res, copy = False, check = False)
                                   
    ##########################################################################
    # nest
    ##########################################################################                               
    def nest(self
             , column_names = None
             , nest_column_name = 'data'
             , include = True
             ):
        '''
        Nest columns of tidy dataframe
        
        Parameters
        ----------
        column_names: str or list of strings
            Columns to be nested
        nest_column_name: str
            Name of the resulting nested column (pandas Series)
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. 'column_names' should not contain all column names.
           
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> 
        >>> penguins_tidy.nest(['bill_length_mm', 'bill_depth_mm',
        >>>                     'flipper_length_mm', 'body_mass_g']
        >>>                    )
        >>> penguins_tidy.nest(['species', 'sex'], include = False)
        '''
        
        cn = self.colnames
        self._validate_column_names(column_names)
        column_names = _enlist(column_names)
        if not include:
            column_names = list(setlist(cn).difference(column_names))
        by = list(setlist(cn).difference(column_names))
        assert len(by) > 0,\
            "arg 'column_names' should not have all existing column names"
            
        return self.nest_by(by = by, nest_column_name = nest_column_name)
        
    ##########################################################################
    # unnest
    ##########################################################################    
    def unnest(self, nest_column_name = 'data'):
        '''
        Unnest a nested column of a tidy dataframe
        
        Parameters
        ----------
        nest_column_name: str
            Name of the column to be unnested
        
        Returns
        -------
        tidyframe
        
        Notes
        -----
        1. unnest does not support a mix of nested lists and dataframe in the
           same column
        
        Examples
        --------
        >>> import pandas as pd
        >>> nested_pdf = pd.DataFrame({"A": [1,2,3, 4],
        >>>                            "B": pd.Series([[10, 20, 30],
        >>>                                            [40, 50],
        >>>                                            [60],
        >>>                                            70
        >>>                                           ]
        >>>                                          )
        >>>                   })
        >>> nested_pdf
        >>> 
        >>> # unnest nested lists
        >>> tidyframe(nested_pdf).unnest('B')
        >>> 
        >>> from palmerpenguins import load_penguins
        >>> penguins = load_penguins().convert_dtypes()
        >>> penguins_tidy = tidyframe(penguins)
        >>> pen_nested_by_species = penguins_tidy.nest_by('species')
        >>> pen_nested_by_species
        >>> 
        >>> # unnest nested dataframes
        >>> pen_nested_by_species.unnest('data')
        '''
        
        nr = self.nrow
        cn = self.colnames
        
        assert nest_column_name in cn,\
            "arg 'nest_column_name' should be a exisiting column name"
            
        all_are_tidy = all(map(lambda x: isinstance(x, tidyframe)
                               , list(self.__data[nest_column_name])
                               )
                           )
        
        def is_list_scalar_na(x):
            if isinstance(x, (list, pd.core.arrays.floating.FloatingArray)):
                res = True
            elif np.isscalar(x):
                res = True
            elif isinstance(pd.isna(x), bool) and pd.isna(x):
                res = True
            else:
                res = False
            return res
        
        all_are_list = all(map(is_list_scalar_na
                               , list(self.__data[nest_column_name])
                               )
                           )
         
        assert all_are_tidy or all_are_list,\
            ("arg 'nest_column_name' column is neither a column of "
             "tidyframes nor a column or lists or scalars"
             )
        
        if all_are_tidy:
            cols = list(setlist(cn).difference([nest_column_name]))
            # create a filled up dataframe per row via cross join and rbind them
            res = pd.concat(
                map(lambda x: pd.merge(
                      self.__data.loc[[x], cols]
                      , (self.__data[nest_column_name][x]).to_pandas(copy = False)
                      , how = 'cross'
                      )
                    , range(nr)
                    )
                , axis = 'index'
                , ignore_index = True
                )
        else:
            res = (self.__data
                       .explode(column = nest_column_name)
                       .reset_index(drop = True)
                       )
                
        return tidyframe(res)
    
    ##########################################################################
    # split (group_split)
    ##########################################################################
    def split(self, by):
        '''
        split
        split rows of a dataframe by groups
        
        Parameters
        ----------
        by: string or a list of strings
            columns to group by
        
        Returns
        -------
        list of tidyframes
        
        Examples
        --------
        >>> from palmerpenguins import load_penguins
        >>> penguins_tidy = tidyframe(load_penguins())
        >>> penguins_tidy.split(by = "species")
        
        
        '''
        
        self._validate_by(by)
        by = _enlist(by)
        cn = self.colnames
        group_cn = _generate_new_string(cn)
        
        res = [x[1] for x in tuple((self.add_group_number(by, name = group_cn)
                                        .__data
                                        .groupby(group_cn
                                                 , dropna = False
                                                 , sort = False
                                                 )
                                        ))]
        res = [(tidyframe(x.reset_index(drop = True), check = False, copy = False)
               .select(group_cn, include = False))
               for x in res
               ]
        return res
    
    group_split = split
    
    ##########################################################################
    # getitem and setitem based on pandas loc
    ##########################################################################
    
    def __getitem__(self, key):
      '''
      Subset some and columns of the dataframe
      Always returns a copy and not a view
      
      Parameters
      ----------
      key: tuple of x and y subset operations
      
      Returns
      -------
      tidyframe
      
      Notes
      -----
      1. Rows can be subset by specifying integer positions 
         (as int, list, slice and range objects) or by providing a boolean mask.
      
      2. Columns can be subset by specifiying integer positions 
         (as int, list, slice and range objects) or by specifying a list of unique 
         column names or by providing a boolean mask.
      
      3. Any combination of row and column specifications work together.
      
      Examples
      --------
      >>> from palmerpenguins import load_penguins
      >>> penguins_tidy = tidyframe(load_penguins())
      >>> 
      >>> # Rows can be subset with integer indexes with slice objects
      >>> # right end is not included in slice objects
      >>> # first four rows
      >>> penguins_tidy[0:4,:]
      >>> 
      >>> # a row can be subset with a single integer
      >>> # moreover subsetting always returns a dataframe
      >>> penguins_tidy[10, :]
      >>> 
      >>> # Rows can be subset with a boolean mask 
      >>> penguins_tidy[penguins_tidy.pull('bill_length_mm') > 40, :]
      >>> 
      >>> # Columns can be subset using column names
      >>> penguins_tidy[0:5, ["species", "year"]]
      >>> 
      >>> # A single column can be subset by specifying column name as a string
      >>> # moreover subsetting always returns a dataframe 
      >>> penguins_tidy[0:5, "species"] # same as: penguins_tidy[0:5, ["species"]]
      >>> 
      >>> # columns can be subset by integer position
      >>> penguins_tidy[[7, 6, 5], 0:3]
      >>> 
      >>> # row and columns can be subset with different types of specifications
      >>> penguins_tidy[0:2, 1] # same as: penguins_tidy[[0, 1], 'island']
      '''
      cn = self.colnames
      key = list(key)
      
      # handle a single integer for row
      if isinstance(key[0], int):
        key[0] = [key[0]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[0], slice):
        key[0] = range(self.nrow)[key[0]]
        
      # handle integer indexing for columns
      is_int = isinstance(key[1], int)
      is_int_list = (isinstance(key[1], list) 
                     and 
                     all([isinstance(x, int) for x in key[1]])
                     )
      if (is_int or is_int_list):
        key[1] = _enlist(key[1])
        assert _is_unique_list(key[1]),\
            "Integer index for columns should be unique"
        assert np.min(key[1]) >= 0,\
            "Integer index for columns should be non-negative"
        assert np.max(key[1]) < self.ncol,\
            "Integer index for columns is beyond the range"
        
        key[1] = [cn[x] for x in key[1]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[1], slice):
        key[1] = cn[key[1]]
      # handle single string column index
      elif isinstance(key[1], str):
        key[1] = _enlist(key[1])
      
      res = (self.to_pandas(copy = False)
                 .loc[key[0], key[1]]
                 .reset_index(drop = True)
                 )
      return tidyframe(res, check = False, copy = True)
    
    def __setitem__(self, key, value):
      '''
      Change a subset of the dataframe in-place
      
      Parameters
      ----------
      key: tuple of x and y subset operations
      value: value to be assigned
      
      Returns
      -------
      tidyframe
      
      Notes
      -----
      1. Rows can be subset by specifying integer positions 
         (as int, list, slice and range objects) or by providing a boolean mask.
      
      2. Columns can be subset by specifiying integer positions 
         (as int, list, slice and range objects) or by specifying a list of unique 
         column names or by providing a boolean mask.
      
      3. Any combination of row and column specifications work together.
      
      4. Assignment is done by "pdf.loc[exp1, exp2] = value". Incompatible value
         assignment exceptions are handled by this method and they will cascade.
      
      Examples
      --------
      >>> from palmerpenguins import load_penguins
      >>> penguins_tidy = tidyframe(load_penguins())
      >>> 
      >>> # assign a single value with correct type
      >>> penguins_tidy[0,0] = "a"
      >>> penguins_tidy[0:5, :]
      >>> 
      >>> # assign a multiple values with correct type
      >>> penguins_tidy[0:3,0] = "b"
      >>> penguins_tidy[0:5, :]
      >>> 
      >>> # assign a row partially by a list of appropriate types
      >>> penguins_tidy[0, ['species', 'bill_length_mm']] = ['c', 1]
      >>> penguins_tidy[0:5, :]
      >>> 
      >>> # assign a subset with another tidyframe
      >>> penguins_tidy[0:2, 0:2] = pd.DataFrame({'species': ['d', 'e']
      >>>                                         , 'island': ['f', 'g']
      >>>                                         }).pipe(tidyframe)
      >>> penguins_tidy[0:5, :]
      '''
      cn = self.colnames
      key = list(key)
      
      # handle a single integer for row
      if isinstance(key[0], int):
        key[0] = [key[0]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[0], slice):
        key[0] = range(self.nrow)[key[0]]
        
      # handle integer indexing for columns
      is_int = isinstance(key[1], int)
      is_int_list = (isinstance(key[1], list) 
               and 
               all([isinstance(x, int) for x in key[1]])
               )
      if (is_int or is_int_list):
        key[1] = _enlist(key[1])
        assert _is_unique_list(key[1]),\
            "Integer index for columns should be unique"
        assert np.min(key[1]) >= 0,\
            "Integer index for columns should be non-negative"
        assert np.max(key[1]) < self.ncol,\
            "Integer index for columns is beyond the range"
        
        key[1] = [cn[x] for x in key[1]]
      # slice should work like regular python (right end is not included)
      elif isinstance(key[1], slice):
        key[1] = cn[key[1]]
      # handle single string column index
      elif isinstance(key[1], str):
        key[1] = _enlist(key[1])
      
      # handle the case when value is a pandas dataframe
      if isinstance(value, (pd.DataFrame
                            , pd.core.groupby.generic.DataFrameGroupBy
                            )):
        msg = ("When arg 'value' is a dataframe, then it should be a "
               "tidyframe and not a pandas dataframe")
        raise Exception(msg)
      
      # handle when value is a tidy pdf
      elif isinstance(value, tidyframe):
        self.to_pandas(copy = False).loc[key[0], key[1]] = value.to_pandas()
      
      else:
        self.to_pandas(copy = False).loc[key[0], key[1]] = value
      # nothing to return as this is an inplace operation
    
    

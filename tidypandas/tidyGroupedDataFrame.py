import copy
import pandas as pd

class TidyGroupedDataFrame:
    
    # init method
    def __init__(self, x, check = True):
        '''
        init
        Create a tidy grouped dataframe from a grouped pandas dataframe

        Parameters
        ----------
        x : pandas grouped dataframe
        check : bool, optional, default is True
            Whether to check if the input pandas grouped dataframe is 'simple'. It is advised to set this to True in user level code.

        Raises
        ------
        Exception if the input pandas grouped dataframe is not 'simple' and warning messages pin point to the precise issue so that user can make necessary changes to the input pandas dataframe. 
        
        Notes
        -----
        A pandas dataframe is said to be 'simple' if:
        1. Column names (x.columns) are an unnamed pd.Index object of unique strings.
        2. Row names (x.index) are an unnamed pd.RangeIndex object with start = 0 and step = 1.
        
        Returns
        -------
        Object of class 'TidyGroupedDataFrame'
        
        Examples
        --------
        from nycflights13 import flights
        flights_grouped_tidy = TidyGroupedDataFrame(flights.groupby('dest'))
        flights_grouped_tidy
        '''
        assert isinstance(check, bool)
        assert isinstance(x, pd.core.groupby.DataFrameGroupBy)
        if check:
            flag_simple = is_simple(x, verbose = True)
            if not flag_simple:    
            # raise the error After informative warnings
                raise Exception(("Input pandas grouped dataframe is not 'simple'."
                                 " See to above warnings."
                                 " Try the 'simplify' function."
                                 " ex: simplify(not simple pandas dataframe) --> simple pandas dataframe."
                                ))
                               
        self.__data = copy.deepcopy(x)   
        return None
    
    # print method
    def __repr__(self):
        header_line     = '-- Tidy grouped dataframe with shape: {shape}'\
        .format(shape = self.__data.obj.shape)
        gvs_line        = "-- Groupby variables: " + str(self.__data.grouper.names)
        n_groups_line   = "-- Number of groups: " + str(self.__data.ngroups)
        few_rows_line   = '-- First few rows:'
        pandas_str_line = self.__data.obj.head(10).__str__()
        
        tidy_string     = (header_line +
                           '\n' +
                           gvs_line +
                           '\n' +
                           n_groups_line +
                           '\n' +
                           few_rows_line +
                           '\n' +
                           pandas_str_line
                           )    
    
        return tidy_string
    
    ##########################################################################
    # to pandas methods
    ##########################################################################
    
    def to_pandas(self):
        '''
        to_pandas
        Returns (a copy) a pandas grouped dataframe

        Returns
        -------
        pandas grouped dataframe
        
        Examples
        --------
        from nycflights13 import flights
        flights_grouped_tidy = TidyGroupedDataFrame(flights.groupby('dest'))
        flights_grouped_tidy.to_pandas()
        '''
        return copy.copy(self.__data)
    
    # series copy method
    def to_series(self, column_name):
        '''
        to_series
        Returns (a copy) a column as grouped pandas series
        
        Parameters
        ----------
        column_name : str
            Name of the column to be returned as grouped pandas series

        Returns
        -------
        pandas series
        
        Examples
        --------
        from nycflights13 import flights
        flights_tidy = TidyDataFrame(flights)
        
        flights_tidy.group_by(['hour', 'day']).to_series('dest').head()
        flights_tidy.group_by(['hour']).to_series('dest').head()
        '''
        assert isinstance(column_name, str)
        assert column_name in self.get_colnames()
        
        gvs = self.get_groupvars()
        # pick column and groupby columns
        ib = (self.select(column_name)
                  .ungroup()
                  .to_pandas()
                  )
        series_obj = ib[column_name]
        
        # create index
        if len(gvs) == 1:
            ind = pd.Index(ib[gvs[0]])
        else:
            ind = pd.MultiIndex.from_frame(ib[gvs])
        
        series_obj.index = ind
        res = series_obj.groupby(series_obj.index)
        return res
        
    
    # to dict method
    def to_dict(self):
        '''
        to_dict
        Returns a dictionary per group

        Returns
        -------
        dict
            A value is a pandas dataframe. Key is a tuple contatining values of grouping columns that define a group.
            
        Examples
        --------
        from nycflights13 import flights
        flights_grouped_tidy = TidyGroupedDataFrame(flights.groupby('dest'))
        flights_grouped_tidy.to_dict()
        '''
        res = dict(tuple(self.__data))
        if len(self.get_groupvars()) == 1:
            keys = list(res.keys())
            for akey in keys:
                res[(akey, )] = res[akey]
                del res[akey]
        
        return res
    
    ##########################################################################
    # pipe methods
    ##########################################################################
    
    def pipe(self, func):
        '''
        pipe
        Returns func(self)

        Parameters
        ----------
        func : callable

        Returns
        -------
        Depends on the return type of `func`
        '''
        return func(self)
    
    # pipe pandas method
    def pipe_pandas(self, func, as_tidy = True):
        '''
        pipe_pandas (alias pipe2)
        Returns func(self.to_pandas())

        Parameters
        ----------
        func : calable
            func should accept the underlying pandas dataframe as input
        as_tidy : bool, optional, default is True
            When True and the result of func(self.to_pandas()) is a pandas dataframe. Then, the result is tidied with tidy_helpers.tidy and converted to a tidyDataframe or a tidyGroupedDataFrame. 

        Returns
        -------
        Depends on the return type of `func`.
        
        Notes
        -----
        Expected usage is when you have to apply a pandas code chunk to a tidy dataframe and convert it back to tidy format.
        
        Examples
        --------
        from nycflights13 import flights
        flights_grouped_tidy = (TidyDataFrame(flights)
                                .group_by('hour')
                                .arrange('dep_time')
                                )
        flights_grouped_tidy
        
        flights_grouped_tidy.pipe2(lambda x: x.head().reset_index(drop = True))
        flights_grouped_tidy.pipe2(lambda x: x.obj.shape)
        '''
        res = func(self.to_pandas())
        if isinstance(res, (pd.DataFrame
                            , pd.core.groupby.DataFrameGroupBy
                            )
                      ):
            res = tidy(res)
        
            if isinstance(res, pd.DataFrame):
                res = TidyDataFrame(res, check = False)
            else:
                res = TidyGroupedDataFrame(res, check = False)
        return res
    
    # alias for pipe_pandas
    pipe2 = pipe_pandas
    
    
    def get_nrow(self):
        '''
        get_nrow
        Get the number of rows
        
        Returns
        -------
        int
        '''
        return self.__data.obj.shape[0]
    
    def get_ncol(self):
        '''
        get_ncol
        Get the number of columns
        
        Returns
        -------
        int
        '''
        return self.__data.obj.shape[1]
        
    def get_shape(self):
        '''
        get_shape (alias get_dim)
        Get the number of rows and columns
        
        Returns
        -------
        tuple
            Number of rows and Number of columns
        '''
        return self.__data.obj.shape
    
    get_dim = get_shape
    
    def get_colnames(self):
        '''
        get_colnames
        Get the column names of the dataframe

        Returns
        -------
        list
            List of unique strings that form the column index of the underlying grouped pandas dataframe
        '''
        return list(self.__data.obj.columns)
        
    def get_groupvars(self):
        '''
        get_groupvars
        Get the grouping column names of the dataframe

        Returns
        -------
        list
            List of unique strings that form the grouping columns of the underlying grouped pandas dataframe
        '''
        return self.__data.grouper.names
    
    ##########################################################################
    # grouby methods
    ##########################################################################
    
    def group_by(self, column_names):
        '''
        group_by
        Group by some column names
        
        Parameters
        ----------
        column_names : str or list of strings
            Names of the columns to be grouped by

        Returns
        -------
        TidyGroupedDataFrame
        
        Notes
        -----
        Tidy grouped dataframes does not remember the order of grouping variables. Grouping variables form a set.
        
        Examples
        --------
        from nycflights13 import flights
        # group by 'dest' and 'origin'
        flights_tidy_grouped = TidyDataFrame(flights).group_by(['dest', 'origin'])
        flights_tidy_grouped
        
        # add 'hour' as a new grouping variable
        flights_tidy_grouped.group_by('hour')
        
        # add 'hour' and 'dest' as grouping variables
        # only 'hour' is the new grouping variable as 'dest' is already a grouping variable
        flights_tidy_grouped.group_by(['hour', 'dest'])
        
        # if you intend to add 'hour' and remove 'dest' from the grouping variables, then use ungroup
        flights_tidy_grouped.ungroup('dest').group_by('hour')
        '''
        
        cns = self.get_colnames()
        assert is_string_or_string_list(column_names),\
            "'column_names' should be a string or list of strings"
        column_names = list(set(enlist(column_names)))
        assert set(cns).issuperset(column_names),\
            "'column_names'(columns to groupby) should be a subset of existing column names"
        
        existing_gvs = self.get_groupvars()
        incoming_gvs = list(set(column_names).difference(existing_gvs))
        if len(incoming_gvs) >= 1:
            print("New grouping columns: " + str(incoming_gvs))
            new_gvs = list(set(existing_gvs).union(incoming_gvs))
            
            res = (self.ungroup()
                       .group_by(new_gvs)
                       )
        else:
            res = copy.copy(self)
        
        return res
    
    # alias for group_by
    groupby = group_by
    
    # ungroup method
    def ungroup(self, column_names = None):
        '''
        ungroup
        Remove some or all grouping columns

        Parameters
        ----------
        column_names : None or str or a list of strings, optional
            When None, all grouping variables are removed and an ungrouped tidy dataframe is returned.
            When str or list of column names are provided, only tose column names are removed from the grouping variables.
                
        Returns
        -------
        TidyDataFrame or TidyGroupedDataFrame
        '''
        
        if column_names is None:
            res = TidyDataFrame(self.__data.obj, check = False)
        else:
            assert is_string_or_string_list(column_names),\
                "`column_names` should be a string or list of strings"
            column_names = enlist(column_names)
            gvs = self.get_groupvars()
            assert set(gvs).issuperset(column_names),\
                "`column_names` (columns to groupby) should be a subset of existing column names"
            new_gvs = list(set(gvs).difference(column_names))
            
            if len(new_gvs) >= 1:
                res = TidyGroupedDataFrame(self.__data.obj.groupby(new_gvs)
                                           , check = False
                                           )
            else:
                res = TidyDataFrame(self.__data.obj, check = False)
        return res
    
    ##########################################################################
    # apply methods
    ##########################################################################   
    
    def group_modify(self, func):
        '''
        group_modify
        Modify dataframe per group

        Parameters
        ----------
        func : callable
            Both input and output of the function should be TidyDataFrame

        Returns
        -------
        TidyGroupedDataFrame
        
        Notes
        -----
        This is a wrapper over apply method of pandas groupby object. 
        
        1. The grouping columns are present in the data chunks provided to `func`. 
        2. If the output per chunk after applying `func` contains any of the grouping columns, then they are removed.
        3. Final result will be grouped by the grouping columns of the input.
        
        Examples
        --------
        from nycflights13 import flights
        ex1 = TidyGroupedDataFrame(flights.groupby('dest'))
        ex1
        
        ex1.group_modify(lambda x: x.slice_head(2))
        ex1.group_modify(lambda x: x.select('origin'))
        
        ex2 = TidyGroupedDataFrame(flights.groupby(['dest', 'origin']))
        ex2
        
        ex2.group_modify(lambda x: x.filter('minute >= 30'))
        '''
        
        gvs = self.get_groupvars()
        
        def func_wrapper(chunk):
            
            res = func(TidyDataFrame(chunk, check = False))
            assert isinstance(res, TidyDataFrame),\
                "`func` should return a tidy ungrouped dataframe"
            res = res.to_pandas()
            
            existing_columns = list(res.columns)
            gvs_in = set(gvs).intersection(existing_columns)
            res = res.drop(columns = list(gvs_in))    
            
            return res
        
        res = (self.__data
                   .apply(func_wrapper)
                   .droplevel(level = len(gvs))
                   .reset_index()
                   .groupby(gvs)
                   )
        
        return TidyGroupedDataFrame(res, check = False)
    
    ##########################################################################
    # basic verbs
    ##########################################################################
    
    ##########################################################################
    # select
    ##########################################################################
    
    def select(self, column_names = None, predicate = None, include = True):
        '''
        select
        Select a subset of columns by name or predicate

        Parameters
        ----------
        column_names : list, optional
            list of column names(strings) to be selected. The default is None.
        predicate : callable, optional
            function which returns a bool. The default is None.
        include : bool, optional
            Whether the columns are to be selected or not. The default is True.

        Returns
        -------
        TidyGroupedDataFrame
        
        Notes
        -----
        1. Select works by either specifying column names or a predicate, not both.
        2. When predicate is used, predicate should accept a pandas series and return a bool. Each column is passed to the predicate and the result indicates whether the column should be selected or not.
        3. When include is False, we select the remaining columns.
        4. Grouping columns are always included in the output.
        
        Examples
        --------
        from nycflights13 import flights
        flights_grouped_tidy = TidyGroupedDataFrame(flights.groupby(['origin', 'dest']))
        flights_grouped_tidy
        
        # select with names
        flights_grouped_tidy.select(['distance', 'hour'])
        
        # select using a predicate: only non-numeric columns
        flights_grouped_tidy.select(predicate = lambda x: x.dtype == "object")
        
        # select columns ending with 'time'
        flights_grouped_tidy.select(predicate = lambda x: bool(re.match(".*time$", x.name)))
        
        # invert the selection
        flights_grouped_tidy.select(['distance', 'hour'], include = False)
        '''
        
        if (column_names is None) and (predicate is None):
            raise Exception('Exactly one among `column_names` and `predicate` should not be None')
        if (column_names is not None) and (predicate is not None):
            raise Exception('Exactly one among `column_names` and `predicate` should not be None')
        
        if column_names is None:
            assert callable(predicate), "`predicate` should be a function"
            col_bool_np = np.array(list(self.__data.obj.apply(predicate, axis = "index")))
            column_names = list(np.array(self.get_colnames())[col_bool_np])
        else:
            assert is_string_or_string_list(column_names),\
                "`column_names` should be a string or list of strings"
            column_names = list(setlist(enlist(column_names)))
            cols = self.get_colnames()
            assert set(cols).issuperset(column_names),\
                "Atleast one string in `column_names` is not an existing column name"
        
        if not include:
            column_names = list(setlist(cols).difference(set(column_names)))
        
        if len(column_names) == 0:
            warnings.warn("None of the columns are selected except grouping columns")
        
        # add group variables
        gvs = self.groupvars()
        column_names = list(set(column_names).union(gvs))
                
        res = (self.__data
                   .obj
                   .loc[:, column_names]
                   .groupby(gvs)
                   )
        return TidyGroupedDataFrame(res, check = False)
        
    
    def relocate(self, column_names, before = None, after = None):
        gvs = self.get_groupvars()
        res = (self.ungroup()
                   .relocate(column_names, before, after)
                   .group_by(gvs)
                   )
        return res
    
    def rename(self, old_new_dict):
        gvs = self.get_groupvars()
        keys = old_new_dict.keys()
        new_gvs = [old_new_dict[agv] if agv in keys else agv for agv in gvs]
        res = (self.ungroup()
                   .rename(old_new_dict)
                   .group_by(new_gvs)
                   )
        return res
         
    def slice(self, row_numbers):
        
        assert all([x >=0 for x in row_numbers])
        groupvars = self.get_groupvars()
        
        res = (self.__data
                   .apply(lambda x: x.iloc[row_numbers, :])
                   .reset_index(drop = True)
                   .groupby(groupvars)
                   )
        
        return TidyGroupedDataFrame(res, check = False)
        
    def arrange(self, column_names, ascending = False, na_position = 'last'):
        
        assert is_string_or_string_list(column_names)
        column_names = enlist(column_names)
        assert len(column_names) > 0
        cn = self.get_colnames()
        assert all([x in cn for x in column_names])
        if not isinstance(ascending, list):
            isinstance(ascending, bool)
        else:
            assert all([isinstance(x, bool) for x in ascending])
            assert len(ascending) == len(column_names)
        
        groupvars = self.get_groupvars()
        res = (self.__data
                   .apply(lambda x: x.sort_values(by = column_names
                                                  , axis         = 0
                                                  , ascending    = ascending
                                                  , inplace      = False
                                                  , kind         = 'quicksort'
                                                  , na_position  = na_position
                                                  , ignore_index = True
                                                  )
                         )
                   .reset_index(drop = True)
                   .groupby(groupvars)
                   )
        
        return TidyGroupedDataFrame(res, check = False)
    
    def filter(self, query_string = None, mask = None):
   
        if query_string is None and mask is None:
            raise Exception("Both 'query' and 'mask' cannot be None")
        if query_string is not None and mask is not None:
            raise Exception("One among 'query' and 'mask' should be None")
        
        group_var_names = self.__data.grouper.names
        
        if query_string is not None and mask is None:
            res = (self.__data
                       .obj
                       .query(query_string)
                       .groupby(group_var_names)
                       )
        if query_string is None and mask is not None:
            res = self.__data.obj.iloc[mask, :]
            res = res.groupby(group_var_names)
            
        res = TidyGroupedDataFrame(res, check = False)
        return res

    def mutate(self, dictionary=None, func=None, column_names = None, predicate = None, prefix = ""):
        if dictionary is None and func is None:
            raise Exception("Either dictionary or func with predicate/column_names should be provided.")

        if dictionary is not None:
            return self._mutate(dictionary)
        else:
            return self._mutate_across(func, column_names=column_names, predicate=predicate, prefix=prefix)

    def _mutate(self, dictionary):
        '''
        {"hp": [lambda x, y: x - y.mean(), ['a', 'b']]
           , "new" : lambda x: x.hp - x.mp.mean() + x.shape[1]
           , "existing" : [lambda x: x + 1]
           }
        TODO:
            2. grouped version
        '''
        assert isinstance(dictionary, dict)
        mutated = copy.deepcopy(self.__data)
        cn = self.get_colnames()

        group_var_names = self.__data.grouper.names

        for akey in dictionary:

            # lambda function case
            if callable(dictionary[akey]):

                def assigner_single(chunk):
                    chunk[akey] = dictionary[akey](chunk)
                    return chunk

                # assigning to single column
                if isinstance(akey, str):
                    mutated = (mutated.apply(assigner_single)
                               .reset_index(drop=True)
                               .groupby(group_var_names)
                               )

                # TODO: assigning to multiple columns
                else:
                    assert all([isinstance(x, str) for x in akey])
                    res_list = dictionary[akey](mutated)
                    assert len(akey) == len(res_list)
                    for i in range(len(akey)):
                        mutated[akey[i]] = res_list[i]

            # simple function case
            if isinstance(dictionary[akey], (list, tuple)):

                # case 1: only simple function
                if len(dictionary[akey]) == 1:
                    assert callable(dictionary[akey][0])

                    # assign to a single column
                    # column should pre-exist
                    if isinstance(akey, str):
                        assert set([akey]).issubset(cn)

                        def assigner_single(chunk):
                            chunk[akey] = dictionary[akey][0](chunk[akey])
                            return chunk

                        mutated = (mutated.apply(assigner_single)
                                   .reset_index(drop=True)
                                   .groupby(group_var_names)
                                   )

                    # TODO: akey is tuple
                    elif isinstance(akey, tuple):
                        assert all([isinstance(x, str) for x in akey])
                        assert set(akey).issubset(cn)
                        input_list = [mutated[colname] for colname in akey]
                        res_list = dictionary[akey][0](*input_list)
                        assert len(akey) == len(res_list)
                        for i in range(len(akey)):
                            mutated[akey[i]] = res_list[i]
                    else:
                        raise TypeError("Column names to be assigned should either be a string or a tuple of strings")
                # case2: simple function with required columns
                elif len(dictionary[akey]) == 2:
                    assert callable(dictionary[akey][0])
                    assert isinstance(dictionary[akey][1], (list, tuple, str))
                    if not isinstance(dictionary[akey][1], (list, tuple)):
                        colnames_to_use = [dictionary[akey][1]]
                    else:
                        colnames_to_use = dictionary[akey][1]
                    assert set(colnames_to_use).issubset(cn)

                    # input_list = [mutated[colname] for colname in colnames_to_use]

                    # assign to a single column
                    if isinstance(akey, str):

                        def assigner_single(chunk):
                            input_cols = map(lambda x: chunk[x], colnames_to_use)
                            chunk[akey] = dictionary[akey][0](*input_cols)
                            return chunk

                        mutated = (mutated.apply(assigner_single)
                                   .reset_index(drop=True)
                                   .groupby(group_var_names)
                                   )

                    # TODO: multiple columns
                    else:
                        res_list = dictionary[akey][0](*input_list)
                        assert len(akey) == len(res_list)
                        for i in range(len(akey)):
                            mutated[akey[i]] = res_list[i]
                else:
                    # TODO create your own error class
                    raise ValueError("Some value(s) in the dictionary is neither callable nor a list or a tuple")

        return TidyGroupedDataFrame(mutated, check=False)
        
    def _mutate_across(self, func, column_names = None, predicate = None, prefix = ""):

        assert callable(func)
        assert isinstance(prefix, str)

        if (column_names is not None) and (predicate is not None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")
        
        if (column_names is None) and (predicate is None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")

        mutated = copy.deepcopy(self.__data)
        grouping_columns = mutated.grouper.names
        cn = mutated.obj.columns

        # use column_names
        if column_names is not None:
            assert isinstance(column_names, list)
            assert all([isinstance(acol, str) for acol in column_names])
        # use predicate to assign appropriate column_names
        else:
            mask_predicate = list(self.__data.obj.apply(predicate, axis=0))
            assert all([isinstance(x, bool) for x in mask_predicate])

            mask_non_grouping_column = [acol not in grouping_columns for acol in cn]

            mask = [x and y for x, y in zip(mask_predicate, mask_non_grouping_column)]
            column_names = cn[mask]
        
        # make a copy of the dataframe and apply mutate in order
        for acol in column_names:
            def assigner_single(chunk):
                chunk[prefix + acol] = func(chunk[acol])
                return chunk
            mutated = (mutated.apply(assigner_single)
                              .reset_index(drop = True)
                              .groupby(grouping_columns))

        return TidyGroupedDataFrame(mutated, check = False)
    
    def summarise(self, dictionary=None, func=None, column_names=None, predicate=None, prefix = ""):
        """

        Parameters
        ----------
        dictionary: dictionary
            map of summarised column names to summarised functions
        func: callable
            aggregate function
        column_names: list
            list of column names(string)
        predicate: callable
            function to select columns, exactly one among column_names and predicate must be specified
        prefix: string
            prefix to add to column names

        Returns
        -------
        a TidyDataFrame

        Note
        -------
        Either dictionary or func with predicate/column names should be provided.

        Examples
        -------
        type1:
        dictionary = {"cross_entropy": [lambda p, q: sum(p*log(q)), ['a', 'b']]}
        dictionary = {"sepal_length": [lambda x: mean(x)]}
        type2:
        dictionary = {"cross_entropy": lambda x: (x.a*log(x.b)).sum()}
        type3:
        dictionary = {"sepal_length": "mean"}

        """
        if dictionary is None and func is None:
            raise Exception("Either dictionary or func with predicate/column_names should be provided.")

        if dictionary is not None:
            return self._summarise(dictionary)
        else:
            return self._summarise_across(func, column_names=column_names, predicate=predicate, prefix=prefix)

    def _summarise(self, dictionary):

        """

        Parameters
        ----------
        dictionary: dictionary
            map of summarised column names to summarised functions

        Returns
        -------
        a TidyDataFrame

        Examples
        -------
        type1:
        dictionary = {"cross_entropy": [lambda p, q: sum(p*log(q)), ['a', 'b']]}
        dictionary = {"sepal_length": [lambda x: mean(x)]}
        type2:
        dictionary = {"cross_entropy": lambda x: (x.a*log(x.b)).sum()}
        type3:
        dictionary = {"sepal_length": "mean"}

        """
        ## 1. will summarised columns earlier in the dictionary be available for later aggregations? NO

        # Type checks
        assert isinstance(dictionary, dict)

        ## don't need a deepcopy as this is data is never overwritten in the method
        grouped = self.__data
        cn = self.get_colnames()

        keys = dictionary.keys()
        values = dictionary.values()

        assert all([isinstance(akey, str) for akey in keys])
        assert all([isinstance(avalue, (list, tuple)) or callable(avalue) for avalue in values])

        # dict: akey -> summarised series
        res = dict()

        for akey in keys:
            avalue = dictionary[akey]
            if isinstance(avalue, (list, tuple)):
                if len(avalue) == 1:
                    func = avalue[0]
                    func_args = [akey]
                elif len(avalue) == 2:
                    func = avalue[0]
                    func_args = avalue[1]
                    assert isinstance(func_args, (list, tuple, str))
                    if isinstance(func_args, str):
                        func_args = [func_args]
                    else:
                        func_args = list(func_args) ## explicitly converts tuples to list
                        assert all(isinstance(acol_name, str) for acol_name in func_args)
                else:
                    raise ValueError("values of type list in the dictionary should be of len 1 or 2")

                assert callable(func)
                assert set(func_args).issubset(cn)
                ## summarise for type 1

                def summariser_single(chunk):
                    input_cols = map(lambda x: chunk[x], func_args)
                    return func(*input_cols)


                res.update({akey: grouped.apply(summariser_single)})

            if callable(avalue):
                ## summarise for type 2
                res.update({akey: grouped.apply(avalue)})

            ## TODO: to support avalue to be a string name for popular aggregate functions.

        list_summarised = list(res.values())

        assert all([a_summarised.shape == list_summarised[0].shape for a_summarised in list_summarised[1:]]), \
            "all summarised series don't have same shape"

        ## importing it here to avoid circular imports
        # from package_tidypandas.TidyDataFrame import TidyDataFrame

        return TidyDataFrame(pd.DataFrame(res).reset_index(drop=False), check=False)

    def _summarise_across(self, func, column_names = None, predicate = None, prefix = ""):
        """

        Parameters
        ----------
        func: callable
            aggregate function
        column_names: list
            list of column names(string)
        predicate: callable
            function to select columns, exactly one among column_names and predicate must be specified
        prefix: string
            prefix to add to column names

        Returns
        -------
        a TidyDataFrame
        """

        assert callable(func)
        assert isinstance(prefix, str)

        if (column_names is not None) and (predicate is not None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")

        if (column_names is None) and (predicate is None):
            raise Exception("Exactly one among 'column_names' and 'predicate' should be None")

        ## don't need a deepcopy as this is data is never overwritten in the method
        grouped = self.__data
        grouping_columns = grouped.grouper.names
        cn = grouped.obj.columns

        # use column_names
        if column_names is not None:
            assert isinstance(column_names, list)
            assert all([isinstance(acol, str) for acol in column_names])
        # use predicate to assign appropriate column_names
        else:
            mask_predicate = list(self.__data.obj.apply(predicate, axis=0))
            assert all([isinstance(x, bool) for x in mask_predicate])

            mask_non_grouping_column = [acol not in grouping_columns for acol in cn]

            mask = [x and y for x, y in zip(mask_predicate, mask_non_grouping_column)]
            column_names = cn[mask]

        # dict: akey -> summarised series
        res = dict()

        for acol in column_names:
            def summariser_single(chunk):
                return func(chunk[acol])
            res.update({prefix+acol: grouped.apply(summariser_single)})

        list_summarised = list(res.values())

        assert all([a_summarised.shape == list_summarised[0].shape for a_summarised in list_summarised[1:]]), \
            "all summarised series don't have same shape"

        ## importing it here to avoid circular imports
        # from package_tidypandas.TidyDataFrame import TidyDataFrame

        return TidyDataFrame(pd.DataFrame(res).reset_index(drop=False), check=False)
    
    def distinct(self, column_names = None, keep = 'first', retain_all_columns = False, ignore_grouping = False):
        
        if column_names is not None:
            assert is_string_or_string_list(column_names)
            column_names = enlist(column_names)
            cols = self.get_colnames()
            assert set(column_names).issubset(cols)
        else:
            column_names = self.get_colnames()
        assert isinstance(retain_all_columns, bool)
        
        
        groupvars = self.get_groupvars()
        cols_subset = set(column_names).difference(groupvars)
        res = (self.__data
                   .apply(lambda x: x.drop_duplicates(subset = cols_subset
                                                      , keep = keep
                                                      , ignore_index = True
                                                      )
                          )
                   .reset_index(drop = True)
                   )
        
        if not retain_all_columns:
            res = res.loc[:, list(set(column_names + groupvars))]
        
        # regroup
        res = res.groupby(groupvars)
        
        return TidyGroupedDataFrame(res, check = False)    
    
    # join methods
    def join(self, y, how = 'inner', on = None, on_x = None, on_y = None, suffix_y = "_y"):

        # assertions
        assert isinstance(y, (TidyDataFrame, TidyGroupedDataFrame))
        assert how in ['inner', 'outer', 'left', 'right', 'anti']
        cn_x = self.get_colnames()
        cn_y = y.get_colnames()
        if isinstance(y, TidyGroupedDataFrame):
            y = y.ungroup().to_pandas()
        else:
            y = y.to_pandas()
            
        if on is None:
            assert on_x is not None and on_y is not None
            assert is_string_or_string_list(on_x)
            assert is_string_or_string_list(on_y)
            on_x = enlist(on_x)
            on_y = enlist(on_y)
            assert len(on_x) == len(on_y)
            assert all([e in cn_x for e in on_x])
            assert all([e in cn_y for e in on_y])
        else: # on is provided
            assert on_x is None and on_y is None
            assert is_string_or_string_list(on)
            on = enlist(on)
            assert all([e in cn_x for e in on])
            assert all([e in cn_y for e in on])
                
        # merge call
        if how == 'anti':
            res = pd.merge(self.ungroup().to_pandas()
                           , y
                           , how = how
                           , on = on
                           , left_on = on_x
                           , right_on = on_y
                           , indicator = True
                           , suffixes = (None, suffix_y)
                           )
            res = res.loc[res._merge == 'left_only', :].drop(columns = '_merge')
        else:    
            res = pd.merge(self.ungroup().to_pandas()
                           , y
                           , how = how
                           , on = on
                           , left_on = on_x
                           , right_on = on_y
                           , suffixes = (None, suffix_y)
                           )
                           
        # remove the new 'on_y' columns
        if on is None:
            def appender(x):
                if x in cn_x:
                    res = x + suffix
                else:
                    res = x
                return res
            
            new_on_y = map(appender, on_y)
            res = res.drop(columns = new_on_y)
        
        # check for unique column names
        res_columns = list(res.columns)
        if len(set(res_columns)) != len(res_columns):
            raise Exception('Join should not result in ambiguous column names. Consider changing the value of "suffix_y" argument')
        
        # bring back the original grouping of x
        groupvars = self.get_groupvars()
        res = res.groupby(groupvars)
                
        return TidyGroupedDataFrame(res, check = False)
        
    def join_inner(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'inner', on, on_x, on_y)
        
    def join_outer(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'outer', on, on_x, on_y)
        
    def join_left(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'left', on, on_x, on_y)
        
    def join_right(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'right', on, on_x, on_y)
        
    def join_anti(self, y, on = None, on_x = None, on_y = None):
        return self.join(y, 'anti', on, on_x, on_y)    
    
    # binding functions
    def cbind(self, y):
        # number of rows should match
        assert self.get_nrow() == y.get_nrow()
        # column names should differ
        assert len(set(self.get_colnames()).intersection(y.get_colnames())) == 0
        
        res = pd.concat([self.ungroup().to_pandas(), y.ungroup().to_pandas()]
                        , axis = 1
                        , ignore_index = False # not to loose column names
                        )
        res = res.groupby(self.get_groupvars())
        return TidyGroupedDataFrame(res, check = False)
    
    def rbind(self, y):
        res = pd.concat([self.ungroup().to_pandas(), y.ungroup().to_pandas()]
                        , axis = 0
                        , ignore_index = True # loose row indexes
                        )
        res = res.groupby(self.get_groupvars())
        return TidyGroupedDataFrame(res, check = False)

    # count
    def count(self, column_names = None, count_column_name = 'n', sort = 'descending'):

        assert (column_names is None) or is_string_or_string_list(column_names)
        if column_names is not None:
            column_names = enlist(column_names)
        assert isinstance(count_column_name, str)
        assert count_column_name not in self.get_colnames()
        assert isinstance(sort, str)
        assert sort in ['asending', 'descending', 'natural']
        
        groupvars = self.get_groupvars()
        if column_names is None:
            column_names = []
            
        temp_groupvars = list(set(column_names + groupvars))
            
        res = (self.ungroup()
                   .to_pandas()
                   .groupby(temp_groupvars)
                   .size()
                   .reset_index()
                   .rename(columns = {0: count_column_name})
                   )
        asc = True
        if sort == 'descending':
            asc = False
        
        if sort != 'natural':
            res = res.sort_values(by = count_column_name
                                  , axis         = 0
                                  , ascending    = asc
                                  , inplace      = False
                                  , kind         = 'quicksort'
                                  , na_position  = 'first'
                                  , ignore_index = True
                                  )
        
        # bring back the grouping
        res = res.groupby(groupvars)

        return TidyGroupedDataFrame(res, check = False)

    def add_count(self
                  , column_names = None
                  , count_column_name = 'n'
                  , sort_order = 'natural'
                  ):


        count_frame = self.count(column_names, count_column_name, sort_order)
        if column_names is None:
            join_names = self.get_groupvars()
        else:
            join_names = list(set(enlist(column_names)).union(self.get_groupvars()))

        res = self.join_inner(count_frame, on = join_names)

        return res
    
    # pivoting
    def pivot_wider(self
                    , names_from
                    , values_from
                    , values_fill = None
                    , values_fn = "mean"
                    , id_cols = None
                    , drop_na = True
                    , retain_levels = False
                    , sep = "__"
                    ):
        
        gvs = self.get_groupvars()
        assert  len(set(enlist(names_from)).union(enlist(values_from)).intersection(gvs)) == 0
        
        if id_cols is not None:
            id_cols = list(set(enlist(id_cols)).union(gvs))
        
        res = (self.ungroup()
                   .pivot_wider(names_from
                                , values_from
                                , values_fill
                                , values_fn
                                , id_cols
                                , drop_na
                                , retain_levels
                                , sep
                                )
                   .group_by(gvs)
                   )
   
        return res
    
    def pivot_longer(self
                     , cols
                     , names_to = "key"
                     , values_to = "value"
                     ):
        
        # assertions
        cn = self.get_colnames()
        assert is_string_or_string_list(cols)
        assert set(cols).issubset(cn)
        
        id_vars = set(cn).difference(cols)
        gvs = self.get_groupvars()
        assert set(gvs).issubset(id_vars)
        assert isinstance(names_to, str)
        assert isinstance(values_to, str)
        assert names_to not in id_vars
        assert values_to not in id_vars
                
        # core operation
        res = (self.ungroup()
                   .pivot_longer(cols        = cols
                                 , names_to  = names_to
                                 , values_to = values_to
                                 )
                   .group_by(gvs)
                   )
        
        return res
    
    # slice extensions
    def slice_head(self, n = None, prop = None):

        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
        
        if prop is None:
            n = int(np.floor(n))
            assert n > 0
            res = (self.__data
                       .apply(lambda chunk : chunk.head(n))
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        else:
            assert prop > 0
            assert prop <= 1
            res = (self.__data
                       .apply(lambda chunk : chunk.head(int(chunk.shape[0] * prop)))
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        
        return TidyGroupedDataFrame(res, check = False)
    
    def slice_tail(self, n = None, prop = None):

        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
        
        if prop is None:
            n = int(np.floor(n))
            assert n > 0
            res = (self.__data
                       .apply(lambda chunk : chunk.tail(n))
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        else:
            assert prop > 0
            assert prop <= 1
            res = (self.__data
                       .apply(lambda chunk : chunk.tail(int(chunk.shape[0] * prop)))
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        
        return TidyGroupedDataFrame(res, check = False)
        
    def slice_sample(self, n = None, prop = None, random_state = None):
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        else:
            assert prop > 0
            assert prop <= 1
            
        if prop is None:
            res = (self.__data
                       .sample(n = n, random_state = random_state, replace = False)
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        else:
            res = (self.__data
                       .sample(frac = prop, random_state = random_state, replace = False)
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        
        return TidyGroupedDataFrame(res, check = False)
    
    def slice_bootstrap(self, n = None, prop = None, random_state = None):
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        else:
            assert prop > 0
            
        if prop is None:
            res = (self.__data
                       .sample(n = n
                               , random_state = random_state
                               , replace = True
                               )
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        else:
            res = (self.__data
                       .sample(frac = prop
                               , random_state = random_state
                               , replace = True
                               )
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        
        return TidyGroupedDataFrame(res, check = False)
    
    def slice_min(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , ties_method = "all"
                  ):
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
            
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            assert prop <= 1
            
        if order_by is None:
            raise Exception("argument 'order_by' should not be None")
        
        if ties_method is None:
            ties_method = "all"
        
        if prop is None:
            res = (self.__data
                       .apply(lambda chunk: chunk.nsmallest(n
                                                            , columns = order_by
                                                            , keep = ties_method
                                                            )
                              )
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        else:
            res = (self.__data
                       .apply(lambda chunk: chunk.nsmallest(int(np.floor(prop * chunk.shape[0]))
                                                            , columns = order_by
                                                            , keep = ties_method
                                                            )
                              )
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        
        return TidyGroupedDataFrame(res, check = False)
    
    def slice_max(self
                  , n = None
                  , prop = None
                  , order_by = None
                  , ties_method = "all"
                  ):
        
        # exactly one of then should be none
        assert not ((n is None) and (prop is None))
        assert not ((n is not None) and (prop is not None))
            
        if n is not None:
            n = int(np.floor(n))
            assert n > 0
        if prop is not None:
            assert prop > 0
            assert prop <= 1
            
        if order_by is None:
            raise Exception("argument 'order_by' should not be None")
        
        if ties_method is None:
            ties_method = "all"
        
        if prop is None:
            res = (self.__data
                       .apply(lambda chunk: chunk.nlargest(n
                                                           , columns = order_by
                                                           , keep = ties_method
                                                           )
                              )
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        else:
            res = (self.__data
                       .apply(lambda chunk: chunk.nlargest(int(np.floor(prop * chunk.shape[0]))
                                                           , columns = order_by
                                                           , keep = ties_method
                                                           )
                              )
                       .reset_index(drop = True)
                       .groupby(self.get_groupvars())
                       )
        
        return TidyGroupedDataFrame(res, check = False)
    
    
    # na handling methods
    def replace_na(self, column_replace_dict):
        gvs = self.get_groupvars()
        res = (self.ungroup()
                   .replace_na(column_replace_dict)
                   .group_by(gvs)
                   )
        return res
    
    def drop_na(self, column_names = None): 
        gvs = self.get_groupvars()
        res = (self.ungroup()
                   .drop_na(column_names)
                   .group_by(gvs)
                   )
        return res
    
    def fill_na(self, column_direction_dict):
        return self.group_modify(lambda chunk : chunk.fill_na(column_direction_dict))
    
    # string methods
    def separate(self
                 , column_name
                 , into
                 , sep = '[^[:alnum:]]+'
                 , strict = True
                 , keep = False
                 ):
        
        gvs = self.get_groupvars()
        assert not column_name in gvs
        
        res = (self.ungroup()
                   .separate(column_name
                             , into = into
                             , sep = sep
                             , strict = strict
                             , keep = keep
                             )
                   .groupby(gvs)
                   )
        
        return res
    
    def unite(self, column_names, into, sep = "_", keep = False):
        
        gvs = self.get_groupvars()
        assert len(set(column_names).intersection(gvs)) == 0
        
        res = (self.ungroup()
                   .unite(column_names, into, sep, keep)
                   .group_by(gvs)
                   )
        
        return res
    
    def separate_rows(self, column_name, sep = ";"):
        
        gvs = self.get_groupvars()
        assert column_name not in gvs
        
        res = (self.ungroup()
                   .separate_rows(column_name, sep)
                   .group_by(gvs)
                   )
        
        return res
    
    def add_rowid(self, column_name = "rowid"):
        
        assert column_name not in self.get_colnames()
        res = self.group_modify(lambda x: x.rowid_to_column(column_name))
        return res
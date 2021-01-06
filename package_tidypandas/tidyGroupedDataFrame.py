import copy
import pandas as pd

class tidyGroupedDataFrame:
    
    # init method
    def __init__(self, x, check = True):
        if check:
            raise TypeError(
                ('If you intend to work with a existing grouped pandas'
                   ' dataframe, then consider removing the grouping structure'
                   ' and creating an instance of tidyDataFrame'
                   ' and then group_by'
                  ))
        else:
            self.__data = copy.copy(x)
        
        return None
    
    # print method
    def __repr__(self):
        print('Tidy grouped dataframe with shape: {shape}'\
               .format(shape = self.__data.obj.shape))
        print("Groupby variables: ", self.__data.grouper.names)
        print("Number of groups: ", self.__data.ngroups)
        print('First few rows:')
        print(self.__data.obj.head(10))
        return ''
    
    # pandas copy method
    def to_pandas(self):
        return copy.copy(self.__data)
    
    # get methods
    def get_info(self):
        print('Tidy grouped dataframe with shape: {shape}'\
               .format(shape = self.__data.obj.shape))
        print("Groupby variables: ", self.__data.grouper.names)
        print("Number of groups: ", self.__data.ngroups)
        print('\n')
        
        return self.__data.obj.info()
    
    def get_nrow(self):
        return self.__data.obj.shape[0]
    
    def get_ncol(self):
        return self.__data.obj.shape[1]
        
    def get_shape(self):
        return self.__data.obj.shape
    
    def get_dim(self):
        return self.__data.obj.shape
    
    def get_colnames(self):
        return list(self.__data.obj.columns)
        
    def get_groupvars(self):
        return self.__data.grouper.names
    
    # ungroup method
    def ungroup(self):
        ## importing it here to avoid circular imports
        from package_tidypandas.tidyDataFrame import tidyDataFrame

        return tidyDataFrame(self.__data.obj, check = False)
     
    # basic verbs   
    def select(self, column_names = None, predicate = None, include = True):
        
        if (column_names is None) and (predicate is None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        if (column_names is not None) and (predicate is not None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        
        if column_names is None:
            assert callable(predicate)
            col_bool_list = list(self.ungroup().to_pandas().apply(predicate))
            column_names = list(np.array(self.get_colnames())[col_bool_list])
            assert len(column_names) > 0
        else:
            assert is_string_or_string_list(column_names)
            column_names = enlist(column_names)
            assert len(column_names) > 0
            cols = self.get_colnames()
            assert all([x in cols for x in column_names])
        
        if not include:
            column_names = set(cols).difference(set(column_names))
            column_names = list(column_names)
            
        groupvars    = self.get_groupvars()
        column_names = list(set(column_names + groupvars))
        
        res = (self.__data.obj.loc[:, column_names]
                              .groupby(groupvars)
                              )
        return tidyGroupedDataFrame(res, check = False)
    
    def slice(self, row_numbers):
        
        assert all([x >=0 for x in row_numbers])
        groupvars = self.get_groupvars()
        
        res = (self.__data
                   .apply(lambda x: x.iloc[row_numbers, :])
                   .reset_index(drop = True)
                   .groupby(groupvars)
                   )
        
        return tidyGroupedDataFrame(res, check = False)
        
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
        
        return tidyGroupedDataFrame(res, check = False)
    
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
            
        res = tidyGroupedDataFrame(res, check = False)
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

        return tidyGroupedDataFrame(mutated, check=False)
        
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

        return tidyGroupedDataFrame(mutated, check = False)
    
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
        a tidyDataFrame

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
        a tidyDataFrame

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
        from package_tidypandas.tidyDataFrame import tidyDataFrame

        return tidyDataFrame(pd.DataFrame(res).reset_index(drop=False), check=False)

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
        a tidyDataFrame
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
        from package_tidypandas.tidyDataFrame import tidyDataFrame

        return tidyDataFrame(pd.DataFrame(res).reset_index(drop=False), check=False)
    
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
        
        return tidyGroupedDataFrame(res, check = False)    
    
    # join methods
    def join(self, y, how = 'inner', on = None, on_x = None, on_y = None, suffix_y = "_y"):

        # assertions
        assert isinstance(y, (tidyDataFrame, tidyGroupedDataFrame))
        assert how in ['inner', 'outer', 'left', 'right', 'anti']
        cn_x = self.get_colnames()
        cn_y = y.get_colnames()
        if isinstance(y, tidyGroupedDataFrame):
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
                
        return tidyGroupedDataFrame(res, check = False)
        
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
        return tidyGroupedDataFrame(res, check = False)

import copy
# from package_tidypandas import tidyDataFrame

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
        return tidyDataFrame(self.__data.obj, check = False)
     
    # basic verbs   
    def select(self, column_names = None, predicate = None, include = True):
        
        if (column_names is None) and (predicate is None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        if (column_names is not None) and (predicate is not None):
            raise Exception('Exactly one among "column_names" and "predicate" should not be None')
        
        if column_names is None:
            assert callable(predicate)
            col_bool_dict = dict(self.__data.apply(predicate))
            for akey in col_bool_dict:
                if not col_bool_dict[akey]:
                    del col_bool_dict[akey]
            column_names = list(col_bool_dict.keys())
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
    
    def mutate(self, dictionary):
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
        cn      = self.get_colnames()
        
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
                                      .reset_index(drop = True)
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
                                          .reset_index(drop = True)
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
                                          .reset_index(drop = True)
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
            
        return tidyGroupedDataFrame(mutated, check = False)
    
    def filter(self, query_string = None, mask = None):
   
        if query_string is None and mask is None:
            raise Exception("Both 'query' and 'mask' cannot be None")
        if query_string is not None and mask is not None:
            raise Exception("One among 'query' and 'mask' should be None")
        
        groupvars = self.get_groupvars()
        
        if query_string is not None and mask is None:
            res = (self.__data
                       .obj
                       .query(query_string)
                       .groupby(groupvars)
                       )
        if query_string is None and mask is not None:
            res = self.__data.obj.iloc[mask, :]
            res = res.groupby(groupvars)
            
        res = tidyGroupedDataFrame(res, check = False)
        return res
    
    def distinct(self, column_names = None, keep = 'first', retain_all_columns = False, ignore_grouping = False):
        
        if column_names is not None:
            assert is_string_or_string_list(column_names)
            column_names = enlist(column_names)
            cols = self.get_colnames()
            assert all([x in cols for x in column_names])
        assert (column_names is None) or (isinstance(column_names, list))
        assert isinstance(retain_all_columns, bool)
        
        groupvars = self.get_groupvars()
        # column_names should not intersect with grouping variables
        if column_names is not None:
            assert not any(x in groupvars for x in column_names)
        
        # function: distinct per chunk
        def distinct_wrapper(chunk):
            
            chunk = chunk.drop(columns = groupvars)
            
            if column_names is None:
                res = chunk.drop_duplicates(keep = keep, ignore_index = True)
            else:
                if retain_all_columns:
                    res = chunk.drop_duplicates(subset = column_names
                                                      , keep = keep
                                                      , ignore_index = True
                                                      )
                else:
                    res = (chunk.loc[:, column_names]
                                .drop_duplicates(keep = keep, ignore_index = True)
                               )
            
            return res
        
        if ignore_grouping:
            if column_names is None:
                res = (self.ungroup()
                           .to_pandas()
                           .drop_duplicates(keep = keep, ignore_index = True)
                           )
            else:
                res = (self.ungroup()
                           .to_pandas()
                           .drop_duplicates(subset = column_names
                                            , keep = keep
                                            , ignore_index = True
                                            )
                            )
            if retain_all_columns:
                res = res.groupby(column_names)
            else:
                if column_names is None:
                    to_select = self.get_colnames()
                else:
                    to_select = list(set(column_names).union(groupvars))
                
                res = (res.loc[:, to_select]
                          .groupby(groupvars)
                          )
        else: # grouped distinct
            res = (self.__data
                       .apply(distinct_wrapper)
                       .reset_index(drop = True, level = 1) # remove col_1
                       .reset_index()
                       .groupby(groupvars)
                       )
        
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
            
        return tidyGroupedDataFrame(res, check = False)

import copy
from package_tidypandas import tidyDataFrame

class tidyGroupedDataFrame:
    def __init__(self, x, check = True):
        if check:
            raise TypeError(
                ('If you want to intend work with a existing grouped pandas'
                   ' dataframe, then consider removing the grouping structure'
                   ' and creating an instance of tidyDataFrame'
                   ' and then group_by'
                  ))
        self.__data = copy.deepcopy(x)
    
    def __repr__(self):
        print('Tidy grouped dataframe with shape: {shape}'\
               .format(shape = self.__data.obj.shape))
        print("Groupby variables: ", self.__data.grouper.names)
        print("Number of groups: ", self.__data.ngroups)
        print('First few rows:')
        print(self.__data.obj.head(10))
        return ''
    
    def to_pandas(self):
        return copy.copy(self.__data)
    
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
        
    def get_colnames(self):
        return list(self.__data.obj.columns)
        
    def select(self, column_names, include = True):
        
        column_names = list(column_names)
        assert len(column_names) == len(set(column_names))
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.__data.obj.columns.to_list()
        assert all([x in cols for x in column_names])
        
        if not include:
            column_names = set(self.__data.obj.columns).difference(set(column_names))
            column_names = list(column_names)
            
        group_var_names = self.__data.grouper.names
        column_names    = list(set(column_names + group_var_names))
        
        res = (self.__data.obj.loc[:, column_names]
                              .groupby(group_var_names)
                              )
        return tidyGroupedDataFrame(res, check = False)
    
    def ungroup(self):
        return tidyDataFrame(self.__data.obj, check = False)
    
    def slice(self, row_numbers):
        
        assert all([x >=0 for x in row_numbers])
        group_var_names = self.__data.grouper.names
        
        res = (self.__data
                   .apply(lambda x: x.iloc[row_numbers, :])
                   .reset_index(drop = True)
                   .groupby(group_var_names)
                   )
        
        return tidyGroupedDataFrame(res, check = False)
        
    def arrange(self, column_names, ascending = False, na_position = 'last'):
        
        if not isinstance(column_names, list):
            column_names = [column_names]
        assert len(column_names) > 0
        cn = self.get_colnames()
        assert all([x in cn for x in column_names])
        if not isinstance(ascending, list):
            isinstance(ascending, bool)
        else:
            assert all([isinstance(x, bool) for x in ascending])
            assert len(ascending) == len(column_names)
        
        group_var_names = self.__data.grouper.names
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
                   .groupby(group_var_names)
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
                        raise TypeError(("Column name(s) to be assigned should ")
                                        ("either be a string or a tuple of strings")
                                        )
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
                    raise ValueError(("Some value(s) in the dictionary is")
                                     ("neither callable nor a list or a tuple")
                                     )
            
        return tidyGroupedDataFrame(mutated, check = False)

import copy

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
        return self.__data
    
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
                   .take(row_numbers)
                   .reset_index()
                   .drop(columns = 'level_' + str(len(group_var_names)))
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
        

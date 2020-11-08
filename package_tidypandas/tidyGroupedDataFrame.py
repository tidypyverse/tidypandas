import copy

class tidyGroupedDataFrame:
    def __init__(self, x):
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
        return tidyGroupedDataFrame(res)
    
    def ungroup(self):
        return tidyDataFrame(self.__data.obj)
    
    def slice(self, row_numbers):
        
        assert all([x >=0 for x in row_numbers])
        group_var_names = self.__data.grouper.names
        
        res = (self.__data
                   .take(row_numbers)
                   .reset_index()
                   .drop(columns = 'level_' + str(len(group_var_names)))
                   .groupby(group_var_names)
                   )
        return tidyGroupedDataFrame(res)

class tidyGroupedDataFrame:
    def __init__(self, x):
        self.data = x
    
    def __repr__(self):
        print('Tidy grouped dataframe with shape: {shape}'\
               .format(shape = self.data.obj.shape))
        print("Groupby variables: ", self.data.grouper.names)
        print("Number of groups: ", self.data.ngroups)
        print('First few rows:')
        print(self.data.obj.head())
        return ''
     
    def info(self):
        print('Tidy grouped dataframe with shape: {shape}'\
               .format(shape = self.data.obj.shape))
        print("Groupby variables: ", self.data.grouper.names)
        print("Number of groups: ", self.data.ngroups)
        print('\n')
        
        return self.data.obj.info()
    
    def select(self, column_names, include = True):
        
        column_names = list(column_names)
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.data.obj.columns.to_list()
        assert all([x in cols for x in column_names])
        
        if not include:
            column_names = self.data.obj.columns.difference(column_names)
        
        group_var_names = self.data.grouper.names
        column_names    = list(set(column_names + group_var_names))
        
        res = (self.data.obj.loc[:, column_names]
                            .groupby(group_var_names)
                            )
        return tidyGroupedDataFrame(res)

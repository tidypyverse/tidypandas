class tidyDataFrame:
    def __init__(self, x):
        self.data = x
    
    def __repr__(self):
        print('Tidy dataframe with shape: {shape}'\
              .format(shape = self.data.shape))
        print('First few rows:')
        print(self.data.head())
        return ''
    
    def info(self):
        print('Tidy dataframe with shape: {shape}'\
              .format(shape = self.data.shape))
        print('\n')
        return self.data.info()
    
    def select(self, column_names, include = True):
        
        column_names = list(column_names)
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.data.columns.to_list()
        assert all([x in cols for x in column_names])
        
        if not include:
            column_names = self.data.columns.difference(column_names)
            if len(column_names) == 0:
                raise "Removing all columns is not allowed"
        
        res = self.data.loc[:, column_names]
        
        return tidyDataFrame(res)
    
    def group_by(self, column_names):
        
        column_names = list(column_names)
        assert len(column_names) > 0
        assert all([isinstance(x, str) for x in column_names])
        cols = self.data.columns.to_list()
        assert all([x in cols for x in column_names])
        
        res = self.data.groupby(column_names)
        return tidyGroupedDataFrame(res)

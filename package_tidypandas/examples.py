import pandas as pd
import numpy as np


iris = pd.read_csv("~/personal/tidypandas/iris.csv")
iris

# ungrouped ----------------------------------------------------------------------

iris_tidy = tidyDataFrame(iris)
iris_tidy

iris_tidy.get_info()
iris_tidy.get_ncol()
iris_tidy.get_nrow()
iris_tidy.get_colnames()
iris_tidy.to_pandas()

iris_tidy.select(['Sepal.Length', 'Species'])
iris_tidy.select(['Sepal.Length', 'Species'], include = False)

iris_tidy.slice([1, 149])

iris_tidy.group_by(['Species'])

iris_tidy.arrange(['Sepal.Length', 'Petal.Width'], ascending = [True, False])

iris_tidy.mutate({"sl" : lambda x:x['Sepal.Length'] + 1})      # standard pandas style
iris_tidy.mutate({"sl" : [lambda x: x + 1, 'Sepal.Length']})   # open single column
iris_tidy.mutate({"sl" : (lambda x: x + 1, ['Sepal.Length'])}) # single column in list
iris_tidy.mutate({"Petal.Length" : (lambda x: x + 1, )})       # assumed to work on key

# all styles can be used within a single call
# mutate executes in order
iris_tidy.mutate({"sl"           : lambda x : x['Sepal.Length'] + x.shape[1],
                  "pl"           : (lambda x: x + 1, 'Petal.Length'),
                  "Petal.Length" : (lambda x: x + 2, )
                 }
                )

# grouped --------------------------------------------------------------------
iris_tidy_grouped = tidyDataFrame(iris).group_by(['Species'])
iris_tidy_grouped

iris_tidy_grouped.get_info()
iris_tidy_grouped.get_ncol()
iris_tidy_grouped.get_nrow()
iris_tidy_grouped.get_colnames()
iris_tidy_grouped.to_pandas()

iris_tidy_grouped.select(['Sepal.Length']) # grouped columns are always kept
iris_tidy_grouped.select(['Sepal.Length', 'Species'], include = False)

iris_tidy_grouped.slice(range(3))

iris_tidy_grouped.ungroup()

iris_tidy_grouped.arrange(['Sepal.Length'], ascending = True).slice([0,1,2])

# notes -----------------------------------------------------------------------
iris.mutate({"Species" : lambda x: fun(x.Species, Sepal.Length)}) # generic
iris.mutate_across(['Species', 'Sepal.Length'], fun) # column name is retained

# sciuba or MP -- no
df.mutate(demeaned = _.hp - _.hp.mean())

# python style lambda -- regular
df.mutate({"hp" : lambda x: x.hp - x.hp.mean()}) # pd.assign
df.mutate({"new" : lambda x: x.hp - x.mp.mean() + x.shape[1] })

# descriptive
df.mutate({"hp": (lambda x, y: x - y.mean(), ["ab", "cd"])})
df.mutate({"hp": (func, ["ab", "cd"])})
df.mutate({"hp": (lambda x, y: x - y.mean(), ["ab", "cd"])
           , "new" : lambda x: x.hp - x.mp.mean() + x.shape[1]
           })
df.mutate({["ab", "cd"] : lambda x: x - x.mean()})
df.mutate({["ab", "cd"]: func})

# descriptive 2
pd.assign(self, a = lambda x: x[a]  - 1)
# next two lines are doable with *kwargs capture
# df.mutate(["ab", "cd"] = lambda x: x - x.mean())
df.mutate(hp = (lambda x, y: x - y.mean(), ['ab', 'cd']))

# implement d2 for intterative work
# implement descriptive for programming

# somehow magically bring data as _ without user asking for it
lambda _, list_of_args: lambda x, y: x - y.mean() + _.shape

def something(**kwargs):
    for k, v in kwargs.items():
        print(k)
        print(v)
        v[0] # function
        v[1] # vars to use
        self[k] = v[0](*[self[i] for i in v[1]])

       
       
something(a = 1,  (a, b) = "c")
    
# final decision on mutate:
df.mutate({"hp": [lambda x, y: x - y.mean(), ['a', 'b']]
           , "new" : lambda x: x.hp - x.mp.mean() + x.shape[1]
           , "existing" : [lambda x: x + 1]
           })
           
f.mutate_across([
    (cols, predictate, fun, prefix = ""),
    (cols, predictate, fun, prefix = "")
])

# similar construct for summarise
# does summarise remove grouping or not: should remove


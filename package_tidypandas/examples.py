import pandas as pd
import numpy as np

iris = pd.read_csv("~/tidypandas/iris.csv")
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
iris_tidy.select(predicate = pd.api.types.is_float_dtype)

iris_tidy.slice([1, 149])

iris_tidy.group_by(['Species'])
iris_tidy.ungroup() # expect a warning

iris_tidy.arrange(['Sepal.Length', 'Petal.Width'], ascending = [True, False])

iris_tidy.mutate({"sl" : lambda x:x['Sepal.Length'] + 1})      # standard pandas style
iris_tidy.mutate({"sl" : [lambda x: x + 1, 'Sepal.Length']})   # open single column
iris_tidy.mutate({"sl" : (lambda x: x + 1, ['Sepal.Length'])}) # single column in list
iris_tidy.mutate({"Petal.Length" : (lambda x: x + 1, )})       # assumed to work on key
iris_tidy.mutate({("sl", "pl") : (lambda x: (x + 1, x + 2), "Sepal.Length")})
iris_tidy.mutate({("Sepal.Length", "Petal.Length") : (lambda x, y: (x + 1, y + 2),)})
iris_tidy.mutate({"sl_plus_pl" : (lambda x, y: x + y, ["Sepal.Length", "Petal.Length"])})

# all styles can be used within a single call
# mutate executes in order
iris_tidy.mutate({"sl"           : lambda x : x['Sepal.Length'] + x.shape[1],
                  "pl"           : (lambda x: x + 1, 'Petal.Length'),
                  "Petal.Length" : (lambda x: x + 2, )
                 }
                )

iris_tidy.mutate(predicate = lambda x: x, func)

iris_tidy.filter("Species == 'setosa'")

iris_tidy.distinct()
iris_tidy.distinct('Species')
iris_tidy.distinct(['Sepal.Length', 'Sepal.Width'])
iris_tidy.distinct(['Sepal.Length', 'Sepal.Width'], retain_all_columns = True)

                               
iris_sepal = (iris_tidy.select(['Sepal.Length', 'Sepal.Width', 'Species']))
iris_petal = (iris_tidy.select(['Petal.Length', 'Petal.Width', 'Species'])) 
iris_sep_pet = (iris_tidy.select(['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Species']))
iris_petal_2 = (iris_petal.mutate({'pl' : (lambda x: x, 'Petal.Length')})
                          .select(['Petal.Length'], include = False)
                          )
                               
# test simple case with on
iris_sepal.join_inner(iris_petal, on = 'Species')
iris_sep_pet.join_inner(iris_petal, on = 'Species')
# with on_x and on_y
iris_petal_2.join_inner(iris_petal, on_x = 'pl', on_y = 'Petal.Length')


# cbind and rbind
iris_sepal.select('Species', include = False).cbind(iris_petal)
iris_sepal.rbind(iris_petal)

iris_tidy.count(count_column_name = "size")
iris_tidy.count('Species')
iris_tidy.count(['Species', 'Sepal.Length'], sort_order = "natural")
iris_tidy.count(['Species', 'Sepal.Length'], sort_order = "descending")
iris_tidy.count(['Species', 'Sepal.Length'], sort_order = "ascending")

iris_tidy.add_count()
iris_tidy.add_count("Species")
iris_tidy.add_count(["Species", "Sepal.Length"])

# pivoting
iris_tidy.pivot_wider(id_cols       = "Sepal.Length"
                      , names_from  = "Species"
                      , values_from = "Petal.Length"
                      )

iris_tidy.pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                      , names_from  = "Species"
                      , values_from = "Petal.Length"
                      )

iris_tidy.pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                      , names_from  = "Species"
                      , values_from = ["Petal.Length", "Petal.Width"]
                      , groupby_id_cols = True
                      )

iris_tidy.pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                      , names_from  = ["Species", "Petal.Width"]
                      , values_from = ["Petal.Length"]
                      )

iris_tidy.pivot_wider(id_cols       = ["Sepal.Length"]
                      , names_from  = ["Species", "Petal.Width"]
                      , values_from = ["Petal.Length", "Sepal.Width"]
                      )

iris_tidy.pivot_wider(names_from  = ["Species", "Petal.Width"]
                      , values_from = ["Petal.Length", "Sepal.Width"]
                      )

# melt
(iris_tidy.pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                      , names_from  = "Species"
                      , values_from = "Petal.Length"
                      )
          .pivot_longer(cols = ["setosa", "versicolor", "virginica"]
                        , names_to = "species"
                        , values_to = "value"
                        )
)

iris_tidy_2 = iris_tidy.mutate({'pl' : (lambda x: x + 1, 'Petal.Length')})
iris_tidy_2

# bind rows and cols
bind_rows([iris_tidy, iris_tidy_2])
bind_cols([iris_tidy, iris_tidy_2])

# slice extensions
iris_tidy.slice_head(3)
iris_tidy.slice_head(151)
iris_tidy.slice_head(prop = 0.1)
iris_tidy.slice_head(prop = 1.1) # should throw an error

iris_tidy.slice_tail(3)
iris_tidy.slice_tail(151)
iris_tidy.slice_tail(prop = 0.1)
iris_tidy.slice_head(prop = 1.1) # should throw an error

iris_tidy.slice_sample(n = 3)
iris_tidy.slice_sample(n = 151)    # should throw an error
iris_tidy.slice_sample(prop = 0.2)
iris_tidy.slice_sample(prop = 1.2) # should throw an error

iris_tidy.slice_bootstrap(n = 20)
iris_tidy.slice_bootstrap(n = 200)
iris_tidy.slice_bootstrap(prop = 0.4)
iris_tidy.slice_bootstrap(prop = 1.4)

iris_tidy.slice_min(n = 3, order_by = 'Sepal.Length')
iris_tidy.slice_max(n = 3, order_by = ['Sepal.Width', 'Sepal.Length']) 

# grouped --------------------------------------------------------------------
iris_tidy_grouped = tidyDataFrame(iris).group_by('Species')
iris_tidy_grouped

iris_tidy_grouped.get_info()
iris_tidy_grouped.get_ncol()
iris_tidy_grouped.get_nrow()
iris_tidy_grouped.get_colnames()
iris_tidy_grouped.to_pandas()

iris_tidy_grouped.select(['Sepal.Length']) # grouped columns are always kept
iris_tidy_grouped.select(['Sepal.Length', 'Species'], include = False)
iris_tidy_grouped.select(predicate = pd.api.types.is_float_dtype)

iris_tidy_grouped.slice(range(2))

iris_tidy_grouped.ungroup() 

iris_tidy_grouped.group_by('Sepal.Length')
iris_tidy_grouped.group_by(['Sepal.Length', 'Species'])

iris_tidy_grouped.arrange('Sepal.Length', ascending = True)
iris_tidy_grouped.mutate(
    {'Sepal.Length' : lambda x: x['Sepal.Length'] + x['Petal.Length'].mean()}
    )

iris_tidy_grouped.mutate(
    {"pl": (lambda x: x + 1, 'Petal.Length')}
    )

iris_tidy_grouped.mutate(
    {"pl": (lambda x, y: x + y, ['Petal.Length', 'Petal.Width'])}
    )
    
iris_tidy_grouped.mutate({"sl"           : lambda x : x['Sepal.Length'] + x.shape[1],
                          "pl"           : (lambda x: x + 1, 'Petal.Length'),
                          "Petal.Length" : (lambda x: x + 2, )
                         }
                         )

iris_tidy_grouped.filter("Species == 'setosa'")

iris_tidy_grouped.distinct()
iris_tidy_grouped.distinct(['Sepal.Length', 'Sepal.Width'])
iris_tidy_grouped.distinct(['Sepal.Length', 'Sepal.Width'], retain_all_columns = True)

# test simple case with on
iris_sepal.group_by('Species').join_inner(iris_petal, on = 'Species')
iris_sepal.join_inner(iris_petal.group_by('Species'), on = 'Species')
iris_sep_pet.group_by('Petal.Length').join_inner(iris_petal, on = 'Species')
# with on_x and on_y
iris_petal_2.group_by('Species').join_inner(iris_petal, on_x = 'pl', on_y = 'Petal.Length')

# cbind and rbind
(iris_sepal.group_by('Sepal.Length')
           .select('Species', include = False)
           .cbind(iris_petal)
           )
(iris_sepal.group_by('Sepal.Length')
           .select('Species', include = False)
           .cbind(iris_petal.group_by('Species'))
           )

(iris_sepal.group_by('Sepal.Length')
           .rbind(iris_petal)
           )

(iris_sepal.group_by('Sepal.Length')
           .rbind(iris_petal.group_by('Species'))
           )

iris_tidy_grouped.count()
iris_tidy_grouped.count('Species')
iris_tidy_grouped.count(['Sepal.Length'])
iris_tidy_grouped.count(['Species', 'Sepal.Length'])

iris_tidy_grouped.add_count()
iris_tidy_grouped.add_count('Species')
iris_tidy_grouped.add_count(['Sepal.Length'])
iris_tidy_grouped.add_count(['Species', 'Sepal.Length'])

# pivoting
iris_tidy_grouped.pivot_wider(id_cols       = "Sepal.Length"
                              , names_from  = "Species"
                              , values_from = "Petal.Length"
                              )

(iris_tidy.group_by("Sepal.Length")
          .pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                       , names_from  = "Species"
                       , values_from = "Petal.Length"
                       )
          )

iris_tidy.pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                      , names_from  = "Species"
                      , values_from = ["Petal.Length", "Petal.Width"]
                      )

iris_tidy.pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                      , names_from  = ["Species", "Petal.Width"]
                      , values_from = ["Petal.Length"]
                      )

iris_tidy.pivot_wider(id_cols       = ["Sepal.Length"]
                      , names_from  = ["Species", "Petal.Width"]
                      , values_from = ["Petal.Length", "Sepal.Width"]
                      )

# melt
(iris_tidy.pivot_wider(id_cols       = ["Sepal.Length", "Sepal.Width"]
                      , names_from  = "Species"
                      , values_from = "Petal.Length"
                      )
          .group_by("Sepal.Length")
          .pivot_longer(cols = ["setosa", "versicolor", "virginica"]
                        , names_to = "species"
                        , values_to = "value"
                        )
)

# slice methods
iris_tidy_grouped.slice_head(n = 2)
iris_tidy_grouped.slice_head(prop = 0.1)

iris_tidy_grouped.slice_tail(n = 3)
iris_tidy_grouped.slice_tail(prop = 0.3)

iris_tidy_grouped.slice_sample(n = 2)
iris_tidy_grouped.slice_sample(prop = 0.05)

iris_tidy_grouped.slice_bootstrap(n = 10)
iris_tidy_grouped.slice_bootstrap(prop = 1.5)

iris_tidy_grouped.slice_min(n = 2, order_by = "Sepal.Length")
iris_tidy_grouped.slice_min(n = 2
                            , order_by = "Sepal.Length"
                            , ties_method = "first"
                            )
iris_tidy_grouped.slice_max(n = 2, order_by = "Sepal.Length")
iris_tidy_grouped.slice_max(n = 2
                            , order_by = "Sepal.Length"
                            , ties_method = "first"
                            )
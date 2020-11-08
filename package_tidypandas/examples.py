import pandas as pd
import numpy as np

iris = pd.read_csv("iris.csv",)
iris

# grouped ----------------------------------------------------------------------

iris_tidy = tidy(iris)
iris_tidy

iris_tidy.get_info()
iris_tidy.get_ncol()
iris_tidy.get_nrow()
iris_tidy.get_colnames()
iris_tidy.to_pandas()

iris_tidy.select(['Sepal.Length', 'Species'])
iris_tidy.select(['Sepal.Length', 'Species'], include = False)

iris_tidy.slice(np.random.choice(np.arange(5), 10))

iris_tidy.group_by(['Species'])

# ungrouped --------------------------------------------------------------------
iris_tidy_grouped = tidy(iris).group_by(['Species'])
iris_tidy_grouped

iris_tidy_grouped.get_info()
iris_tidy_grouped.get_ncol()
iris_tidy_grouped.get_nrow()
iris_tidy_grouped.get_colnames()
iris_tidy_grouped.to_pandas()

iris_tidy_grouped.select(['Sepal.Length']) # grouped columns are always kept
iris_tidy_grouped.select(['Sepal.Length', 'Species'], include = False)

iris_tidy_grouped.slice(np.random.choice(np.arange(5), 10))

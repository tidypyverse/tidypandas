from package_tidypandas.tidyDataFrame import *
from package_tidypandas.tidypredicates import *
import seaborn as sns


def test_mutate_across():
    iris = sns.load_dataset("iris")
    iris_tidy = tidyDataFrame(iris)
    iris_tidy = iris_tidy.mutate_across(lambda x: x - x.mean(), predicate=ends_with("length"), prefix="centered_")
    assert all(iris_tidy.to_pandas()["centered_sepal_length"].eq(iris.sepal_length - iris.sepal_length.mean()))

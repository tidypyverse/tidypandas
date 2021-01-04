from package_tidypandas.tidyDataFrame import *
from package_tidypandas.tidyGroupedDataFrame import *
from package_tidypandas.tidypredicates import *
import seaborn as sns



def test_mutate_with_dict():
    iris = sns.load_dataset("iris")
    iris_tidy = tidyDataFrame(iris)

    ## Tidy way
    iris_tidy = iris_tidy.mutate(
                                    {"sl": lambda x: x['sepal_length'] + x.shape[1],
                                      "pl": (lambda x: x + 1, 'petal_length'),
                                      "petal_length": (lambda x: x + 2,)
                                    }
                                )

    ## pandas way
    sl = iris["sepal_length"] + iris.shape[1]
    pl = iris["petal_length"] + 1
    petal_length = iris["petal_length"] + 2

    ## assertions
    assert all((iris_tidy.to_pandas()["sl"] - sl) < 0.0001)
    assert all((iris_tidy.to_pandas()["pl"] - pl) < 0.0001)
    assert all((iris_tidy.to_pandas()["petal_length"] - petal_length) < 0.0001)

def test_mutate_with_predicate():
    iris = sns.load_dataset("iris")
    iris_tidy = tidyDataFrame(iris)
    iris_tidy = iris_tidy.mutate(func=lambda x: x - x.mean(),
                                 predicate=lambda x: x.name[-6:] == "length",
                                 prefix="centered_")
    assert all(iris_tidy.to_pandas()["centered_sepal_length"]
               .eq(iris.sepal_length - iris.sepal_length.mean()))

def test_summarise_with_dict():
    iris = sns.load_dataset("iris")
    iris_tidy = tidyDataFrame(iris)

    ## Tidy way
    summarised_iris = \
        iris_tidy. \
            summarise(
            {
                "sepal_length_adj_1": (lambda x, y: ((x - y.mean()) / x.std()).sum(),
                                       ["sepal_length", "petal_length"]),
                "sepal_length_adj_2":
                    lambda x: (
                                (x.sepal_length - x.petal_length.mean())
                                / x.sepal_length.std()
                             ).sum(),
                "sepal_width": [lambda x: ((x - x.mean()) ** 2).mean()]
            }
        )

    ## pandas way
    sepal_length_adj_1 = (
                            (iris["sepal_length"] - iris["petal_length"].mean())\
                            /(iris["sepal_length"].std())
                          ).sum()

    sepal_width_var = ((iris["sepal_width"] - iris["sepal_width"].mean()) ** 2).mean()

    ## assertions
    assert summarised_iris.get_nrow() == 1
    assert all(summarised_iris.to_pandas()["sepal_length_adj_1"] - sepal_length_adj_1 < 0.0001)
    assert all(summarised_iris.to_pandas()["sepal_length_adj_2"] - sepal_length_adj_1 < 0.0001)
    assert all(summarised_iris.to_pandas()["sepal_width"] - sepal_width_var < 0.0001)


def test_summarise_with_predicate():
    iris = sns.load_dataset("iris")

    ## Tidy way
    iris_tidy = tidyDataFrame(iris)

    summarised_iris = iris_tidy.summarise(
        func=np.mean,
        predicate=lambda x: x.name[-6:] == "length",
        prefix="mean_"
    )

    ## Pandas way
    sepal_length_mean = iris["sepal_length"].mean()
    petal_length_mean = iris["petal_length"].mean()

    ## assertions
    assert summarised_iris.get_nrow() == 1
    assert all((summarised_iris.to_pandas()["mean_sepal_length"] - sepal_length_mean) < 0.0001)
    assert all((summarised_iris.to_pandas()["mean_petal_length"] - petal_length_mean) < 0.0001)

def test_grouped_mutate_with_dict():
    iris = sns.load_dataset("iris")

    ## Tidy way
    iris_tidy = tidyDataFrame(iris)
    iris_tidy_grouped = iris_tidy.group_by("species")
    iris_tidy_grouped = \
        iris_tidy_grouped.mutate({"sl": lambda x: x['sepal_length'] + x.shape[1],
                                  "pl": (lambda x: x - np.mean(x), 'petal_length'),
                                  "petal_length": (lambda x: x - np.mean(x),)
                                 })

    ## Pandas way
    iris_grouped = iris.groupby("species")
    sl = iris_grouped.apply(lambda x: x["sepal_length"] + x.shape[1]).reset_index(drop=True)
    pl = iris_grouped["petal_length"].apply(lambda x: x-x.mean()).reset_index(drop=True)

    ## assertions
    assert all((iris_tidy_grouped.ungroup().to_pandas()["sl"] - sl) < 0.0001)
    assert all((iris_tidy_grouped.ungroup().to_pandas()["pl"] - pl) < 0.0001)
    assert all((iris_tidy_grouped.ungroup().to_pandas()["petal_length"] - pl) < 0.0001)


def test_grouped_mutate_with_predicate():
    iris = sns.load_dataset("iris")

    ## Tidy way
    iris_tidy = tidyDataFrame(iris)
    iris_tidy_grouped = iris_tidy.group_by("species")
    iris_tidy_grouped = iris_tidy_grouped.mutate(
        func=lambda x: x - x.mean(),
        predicate=lambda x: x.name[-6:] == "length",
        prefix="centered_"
    )

    ## Pandas way
    iris_grouped = iris.groupby("species")
    sepal_length_centered = iris_grouped["sepal_length"].apply(lambda x: x-x.mean()).reset_index(drop=True)
    petal_length_centered = iris_grouped["petal_length"].apply(lambda x: x-x.mean()).reset_index(drop=True)

    ## assertions
    assert all((iris_tidy_grouped.ungroup().to_pandas()["centered_sepal_length"] - sepal_length_centered) < 0.0001)
    assert all((iris_tidy_grouped.ungroup().to_pandas()["centered_petal_length"] - petal_length_centered) < 0.0001)


def test_grouped_summarise_with_dict():
    iris = sns.load_dataset("iris")
    iris_tidy = tidyDataFrame(iris)

    ## Tidy way
    iris_tidy_grouped = iris_tidy.group_by("species")
    summarised_iris = \
        iris_tidy_grouped. \
            summarise(
            {
                "sepal_length_adj_1": (lambda x, y: ((x - y.mean()) / x.std()).sum(),
                                       ["sepal_length", "petal_length"]),
                "sepal_length_adj_2":
                    lambda x: (
                            (x.sepal_length - x.petal_length.mean())
                            / x.sepal_length.std()
                    ).sum(),
                "sepal_width": [lambda x: ((x - x.mean()) ** 2).mean()]
            }
        )

    ## pandas way
    iris_grouped = iris.groupby("species")
    sepal_length_mean = iris_grouped["sepal_length"].mean()
    sepal_length_std = iris_grouped["sepal_length"].std()
    petal_length_mean = iris_grouped["petal_length"].mean()
    sepal_width_mean = iris_grouped["sepal_width"].mean()
    stats_frame = pd.DataFrame({"sepal_length_mean": sepal_length_mean,
                                "sepal_length_std": sepal_length_std,
                                "petal_length_mean": petal_length_mean,
                                "sepal_width_mean": sepal_width_mean})
    iris = iris.join(stats_frame, how="left", on=["species"])

    sepal_length_adj_1 = ((iris["sepal_length"] - iris["petal_length_mean"])
                          / (iris["sepal_length_std"])). \
        groupby(iris["species"]).sum()
    sepal_width_var = ((iris["sepal_width"] - iris["sepal_width_mean"]) ** 2).groupby(iris["species"]).mean()

    assert all(summarised_iris.to_pandas()["sepal_length_adj_1"] - sepal_length_adj_1.reset_index(drop=True) < 0.0001)
    assert all(summarised_iris.to_pandas()["sepal_length_adj_2"] - sepal_length_adj_1.reset_index(drop=True) < 0.0001)
    assert all(summarised_iris.to_pandas()["sepal_width"] - sepal_width_var.reset_index(drop=True) < 0.0001)


def test_grouped_summarise_with_predicate():
    iris = sns.load_dataset("iris")

    ## Tidy way
    iris_tidy = tidyDataFrame(iris)
    iris_tidy_grouped = iris_tidy.group_by("species")

    summarised_iris = iris_tidy_grouped.summarise(
        func=np.mean,
        predicate=lambda x: x.name[-6:] == "length",
        prefix="mean_"
    )

    ## Pandas way
    iris_grouped = iris.groupby("species")
    sepal_length_mean = iris_grouped["sepal_length"].mean().reset_index(drop=True)
    petal_length_mean = iris_grouped["petal_length"].mean().reset_index(drop=True)

    ## assertions
    assert all((summarised_iris.to_pandas()["mean_sepal_length"] - sepal_length_mean) < 0.0001)
    assert all((summarised_iris.to_pandas()["mean_petal_length"] - petal_length_mean) < 0.0001)
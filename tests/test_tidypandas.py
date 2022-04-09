import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from pandas._testing import assert_frame_equal
from pandas.api import types
import pytest

from tidypandas import tidyframe
from tidypandas.tidy_utils import simplify

@pytest.fixture
def penguins_data():
    return simplify(load_penguins())

def assert_frame_equal_v2(a, b):
    ## column order independent assert equal for dataframes
    return assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)])


def test_mutate(penguins_data):

    penguins_tidy = tidyframe(penguins_data, copy=False)
    # Ungrouped
    # Type 1
    expected = penguins_data.assign(ind=np.arange(344))
    result = penguins_tidy.mutate({'ind' : np.arange(344)}).to_pandas(copy=False)
    assert_frame_equal_v2(result, expected)

    # Type 2
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": lambda x: x['year'] + 1}).to_pandas()
    assert_frame_equal_v2(result, expected)

    # Type 3
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": "x['year'] + 1"}).to_pandas()
    assert_frame_equal_v2(result, expected)

    # Type 4
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": (lambda x: x + 1, "year")}).to_pandas()
    assert_frame_equal_v2(result, expected)

    # Type 5
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": ("x + 1", "year")}).to_pandas()
    assert_frame_equal_v2(result, expected)

    # Type 6
    expected = penguins_data.assign(x_plus_y=lambda x: x["bill_length_mm"]+x["bill_depth_mm"])
    result = penguins_tidy.mutate({"x_plus_y": (lambda x, y: x + y, ["bill_length_mm", "bill_depth_mm"])}).to_pandas()
    assert_frame_equal_v2(result, expected)

    # Type 7 
    expected = penguins_data.assign(x=lambda x: x["year"]-1).assign(y=lambda x: x["year"]+1)
    result = penguins_tidy.mutate({('x', 'y'): lambda x: [x['year'] - 1, x['year'] + 1]}).to_pandas()
    assert_frame_equal_v2(result, expected)

    # Type 8
    expected = penguins_data.assign(bill_length_mm=lambda x: x["bill_length_mm"]+1)
    result = penguins_tidy.mutate({'bill_length_mm': ("x + 1", )}).to_pandas()    
    assert_frame_equal_v2(result, expected)    

    ## With kwargs
    expected = penguins_data[["year"]].assign(yp1=lambda x: x["year"]+10)
    result = (penguins_tidy.select('year')
                           .mutate({'yp1': ("x + kwargs['akwarg']", 'year')}, akwarg = 10)).to_pandas()
    assert_frame_equal_v2(expected, result)


def test_mutate_groupby(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)


    expected = penguins_data.groupby("sex", dropna=False)\
                  .apply(lambda x: x.assign(year_mod=lambda y: y["year"] + np.mean(y["year"]) - 4015))
    result = penguins_tidy.mutate({'year_mod' : "x['year'] + np.mean(x['year']) - 4015"}
                                   , by = 'sex'
                                   ).to_pandas()
    assert_frame_equal_v2(expected, result)

def test_mutate_orderby(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)

    expected = (penguins_data[['year', 'species', 'bill_length_mm']]\
                        .sort_values("bill_length_mm")\
                        .assign(year_cumsum=lambda x: np.cumsum(x["year"]))\
                        .sort_index())

    result = (penguins_tidy.select(['year', 'species', 'bill_length_mm'])
                           .mutate({'year_cumsum': (np.cumsum, 'year')},
                                      order_by = 'bill_length_mm'
                                      ).to_pandas()
                )
    assert_frame_equal_v2(expected, result)


def test_mutate_across(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)



    # across mode with column names
    expected = penguins_data[['bill_length_mm', 'body_mass_g']]\
                    .assign(demean_bill_length_mm=lambda x: x["bill_length_mm"] - np.mean(x["bill_length_mm"]))\
                    .assign(demean_body_mass_g=lambda x: x["body_mass_g"] - np.mean(x["body_mass_g"]))
    result = (penguins_tidy.select(['bill_length_mm', 'body_mass_g'])
                  .mutate(column_names = ['bill_length_mm', 'body_mass_g']
                          , func = lambda x: x - np.mean(x)
                          , prefix = "demean_"
                          )
                  ).to_pandas()
    assert_frame_equal_v2(expected, result)
                  
    # grouped across with column names
    expected = penguins_data[['bill_length_mm', 'body_mass_g', 'species']]
    expected = expected.groupby("species")\
                .apply(lambda x: x.assign(demean_bill_length_mm=lambda y: y["bill_length_mm"] - np.mean(y["bill_length_mm"]))\
                                  .assign(demean_body_mass_g=lambda y: y["body_mass_g"] - np.mean(y["body_mass_g"])))

    result = (penguins_tidy.select(['bill_length_mm', 'body_mass_g', 'species'])
                           .mutate(column_names = ['bill_length_mm', 'body_mass_g'],
                                    func = lambda x: x - np.mean(x),
                                    prefix = "demean_",
                                    by = 'species'
                                    )
              ).to_pandas()
    assert_frame_equal_v2(expected, result)


def test_summarise(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)

    expected = pd.DataFrame({"A": [np.mean(penguins_data["bill_length_mm"] + penguins_data["bill_depth_mm"])]
                            , "B": [np.mean(penguins_data["bill_length_mm"] - penguins_data["bill_depth_mm"])]})
    result = penguins_tidy.summarise({('A', 'B'): ("[np.mean(x + y), np.mean(x - y)]"
                                                      , ["bill_length_mm", "bill_depth_mm"]
                                                      )}
                                    ).to_pandas()
    assert_frame_equal_v2(expected, result)

    # summarise across
    expected = pd.DataFrame({"bill_length_mm": [np.mean(penguins_data["bill_length_mm"])]
                             , "bill_depth_mm": [np.mean(penguins_data["bill_depth_mm"])]
                             }
                             )
    result = penguins_tidy.summarise(
                                func = np.mean,
                                column_names = ['bill_length_mm', 'bill_depth_mm']
                                ).to_pandas()
    assert_frame_equal_v2(expected, result)

def test_summarise_groupby(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)


    # expected = penguins_data.groupby(["species", "sex"], dropna=False)\
    #                         .apply(lambda x: x[["year"]].mean().rename({"year": "a_mean"}))\
    #                         .reset_index().sort_values(["species", "sex"])
    expected = penguins_data.groupby(["species", "sex"], dropna=False)\
                            .apply(lambda x: np.mean(x["year"]))\
                            .reset_index().rename(columns={0: "a_mean"})\
                            .sort_values(["species", "sex"]).reset_index(drop=True)
    expected = expected.convert_dtypes()


    result = penguins_tidy.summarise({"a_mean": (np.mean, 'year')},
                                      by = ['species', 'sex']
                                      ).to_pandas()\
                                       .sort_values(["species", "sex"]).reset_index(drop=True)
    assert_frame_equal_v2(expected, result)


    # groupby summarise with kwargs
    expected = penguins_data.groupby(["species", "sex"], dropna=False)\
                            .apply(lambda x: np.mean(x["year"]) + 4)\
                            .reset_index().rename(columns={0: "a_mean"})\
                            .sort_values(["species", "sex"]).reset_index(drop=True)
    expected = expected.convert_dtypes()

    result = penguins_tidy.summarise(
                              {"a_mean": lambda x, **kwargs: x['year'].mean() + kwargs['leap']}
                              , by = ['species', 'sex']
                              , leap = 4
                            ).to_pandas().sort_values(["species", "sex"]).reset_index(drop=True)
    assert_frame_equal_v2(expected, result)

    # groupby summarise across
    expected = penguins_data.groupby(["species", "sex"], dropna=False)\
                            .apply(lambda x: x.loc[:, x.columns[x.apply(types.is_numeric_dtype, axis=0)]].mean()\
                                              .add_prefix("avg_"))\
                            .reset_index()\
                            .sort_values(["species", "sex"]).reset_index(drop=True)
    expected = expected.convert_dtypes()

    result = penguins_tidy.summarise(
                                      func = np.mean,
                                      predicate = types.is_numeric_dtype,
                                      prefix = "avg_",
                                      by = ['species', 'sex']
                                    ).to_pandas().sort_values(["species", "sex"]).reset_index(drop=True)
    assert_frame_equal_v2(expected, result)


def test_select(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)

    sel_cols = penguins_data.convert_dtypes().columns[penguins_data.convert_dtypes().apply(lambda x: x.dtype != "string", axis=0)]
    expected = penguins_data.loc[:, sel_cols]
    result = penguins_tidy.select(predicate = lambda x: x.dtype != "string").to_pandas()
    assert_frame_equal_v2(expected, result)

    sel_cols = list(set(penguins_data.columns).difference(set(['sex', 'species'])))
    expected = penguins_data.loc[:, sel_cols]
    result = penguins_tidy.select(['sex', 'species'], include = False).to_pandas()
    assert_frame_equal_v2(expected, result)

def test_filter(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)
    pass

## test joins

## test pivot





  









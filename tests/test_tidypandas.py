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

    result = penguins_tidy.summarise(
                                      func = np.mean,
                                      predicate = types.is_numeric_dtype,
                                      prefix = "avg_",
                                      by = ['species', 'sex']
                                    ).to_pandas().sort_values(["species", "sex"]).reset_index(drop=True)
    assert_frame_equal_v2(expected, result)


def test_select(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)

    sel_cols = penguins_data.columns[penguins_data.apply(lambda x: x.dtype != "string", axis=0)]
    expected = penguins_data.loc[:, sel_cols]
    result = penguins_tidy.select(predicate = lambda x: x.dtype != "string").to_pandas()
    assert_frame_equal_v2(expected, result)

    sel_cols = list(set(penguins_data.columns).difference(set(['sex', 'species'])))
    expected = penguins_data.loc[:, sel_cols]
    result = penguins_tidy.select(['sex', 'species'], include = False).to_pandas()
    assert_frame_equal_v2(expected, result)

def test_filter(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)
    exp = penguins_tidy.filter(lambda x: x['bill_length_mm'] >= x['bill_length_mm'].mean(), by = 'species')
    assert isinstance(exp, tidyframe)
    
    # string test
    exp = penguins_tidy.filter('bill_length_mm > 35', by = 'species')
    assert isinstance(exp, tidyframe)
    
## test joins
def test_joins(penguins_data):
    penguins_tidy = tidyframe(penguins_data, copy=False)

    # inner join ---------------------------------------------------------------
    penguins_tidy_s1 = (penguins_tidy.tail(n = 1, by = 'species')
                                     .select(['species', 'bill_length_mm', 'island'])
                                     )
    penguins_tidy_s2 = (penguins_tidy.head(n = 1, by = 'species')
                                     .select(['species', 'island', 'bill_depth_mm'])
                                     )
    penguins_tidy_s3 = penguins_tidy_s2.rename({'island': 'island2'})
    
    # on-test                              
    res = penguins_tidy_s1.inner_join(penguins_tidy_s2,
                                      on = 'island',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm', 'island', 'species_y','bill_depth_mm']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # on_x and on_y test
    res = penguins_tidy_s1.inner_join(penguins_tidy_s3,
                                      on_x   = 'island',
                                      on_y   = 'island2',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm',
                     'island', 'species_y','bill_depth_mm', 'island2']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # left join ---------------------------------------------------------------
    # on-test                              
    res = penguins_tidy_s1.left_join(penguins_tidy_s2,
                                      on = 'island',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm', 'island', 'species_y','bill_depth_mm']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # on_x and on_y test
    res = penguins_tidy_s1.left_join(penguins_tidy_s3,
                                      on_x   = 'island',
                                      on_y   = 'island2',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm',
                     'island', 'species_y','bill_depth_mm', 'island2']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # right join ---------------------------------------------------------------
    # on-test                              
    res = penguins_tidy_s1.right_join(penguins_tidy_s2,
                                      on = 'island',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm', 'island', 'species_y','bill_depth_mm']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # on_x and on_y test
    res = penguins_tidy_s1.right_join(penguins_tidy_s3,
                                      on_x   = 'island',
                                      on_y   = 'island2',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm',
                     'island', 'species_y','bill_depth_mm', 'island2']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # full join ---------------------------------------------------------------
    # on-test                              
    res = penguins_tidy_s1.full_join(penguins_tidy_s2,
                                      on = 'island',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm', 'island', 'species_y','bill_depth_mm']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # on_x and on_y test
    res = penguins_tidy_s1.full_join(penguins_tidy_s3,
                                      on_x   = 'island',
                                      on_y   = 'island2',
                                      suffix = ['_x', '_y']
                                      )
    exp_cols_list = ['species_x', 'bill_length_mm',
                     'island', 'species_y','bill_depth_mm', 'island2']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # semi join ---------------------------------------------------------------
    # on-test                              
    res = penguins_tidy_s1.semi_join(penguins_tidy_s2,on = 'island')
    exp_cols_list = ['species', 'bill_length_mm', 'island']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # on_x and on_y test
    res = penguins_tidy_s1.semi_join(penguins_tidy_s3,
                                      on_x   = 'island',
                                      on_y   = 'island2'
                                      )
    exp_cols_list = ['species', 'bill_length_mm','island']
    print(sorted(res.colnames))
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # anti join ---------------------------------------------------------------
    # on-test                              
    res = penguins_tidy_s1.anti_join(penguins_tidy_s2,on = 'island')
    exp_cols_list = ['species', 'bill_length_mm', 'island']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # on_x and on_y test
    res = penguins_tidy_s1.anti_join(penguins_tidy_s3,
                                      on_x   = 'island',
                                      on_y   = 'island2'
                                      )
    exp_cols_list = ['species', 'bill_length_mm','island']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
    # cross join ---------------------------------------------------------------
    # on-test                              
    res = penguins_tidy_s1.cross_join(penguins_tidy_s2)
    exp_cols_list = ['species', 'bill_length_mm', 'island',
                     'species_y', 'island_y', 'bill_depth_mm']
    assert sorted(res.colnames) == sorted(exp_cols_list)
    
## test pivot

## test slice and its extensions
def test_slice(penguins_data):
    penguins_tidy = tidyframe(penguins_data)
    
    # simple slice
    exp = penguins_data.iloc[range(2, 5), :].reset_index(drop = True)
    res = penguins_tidy.slice(range(2, 5)).to_pandas()
    assert_frame_equal_v2(exp, res)

    # grouped slice
    exp = (penguins_data.groupby('species')
                        .apply(lambda x: x.iloc[range(2, 5), :])
                        .reset_index(drop = True)
                        )
    res = (penguins_tidy.slice(range(2, 5), by = 'species')
                        .arrange('species')
                        .to_pandas()
                        )
    assert_frame_equal_v2(exp, res)
    
    # simple slice head
    exp = (penguins_data.iloc[range(3), :]
                        .reset_index(drop = True)
                        )
                        
    res = penguins_tidy.slice_head(n = 3).to_pandas()
    assert_frame_equal_v2(exp, res)
    
    # grouped slice head
    exp = (penguins_data.groupby('species')
                        .apply(lambda x: x.iloc[range(3), :])
                        .reset_index(drop = True)
                        )
                        
    res = (penguins_tidy.slice_head(n = 3, by = 'species')
                        .arrange('species')
                        .to_pandas()
                        )
    assert_frame_equal_v2(exp, res)
    
    # simple slice tail
    exp = (penguins_data.tail(3)
                        .reset_index(drop = True)
                        )
                        
    res = penguins_tidy.slice_tail(n = 3).to_pandas()
    assert_frame_equal_v2(exp, res)
    
    # grouped slice tail
    exp = (penguins_data.groupby('species')
                        .apply(lambda x: x.tail(3))
                        .reset_index(drop = True)
                        )
                        
    res = (penguins_tidy.slice_tail(n = 3, by = 'species')
                        .arrange('species')
                        .to_pandas()
                        )
    assert_frame_equal_v2(exp, res)
  
    # simple slice_max
    exp = (penguins_data.nlargest(3, 'bill_length_mm')
                        .sort_values('bill_length_mm')
                        .reset_index(drop = True)
                        )
                        
    res = (penguins_tidy.slice_max(n = 3, order_by_column = 'bill_length_mm', with_ties = False)
                        .arrange('bill_length_mm')
                        .to_pandas()
                        )
    assert_frame_equal_v2(exp, res)
    
    # grouped slice_max
    exp = (penguins_data.groupby('species')
                        .apply(lambda x: x.nlargest(3, 'bill_length_mm'))
                        .reset_index(drop = True)
                        .sort_values('bill_length_mm', ignore_index = True)
                        )
                        
    res = (penguins_tidy.slice_max(n = 3, order_by_column = 'bill_length_mm', by = 'species', with_ties = False)
                        .arrange('bill_length_mm')
                        .to_pandas()
                        )
    assert_frame_equal_v2(exp, res)

    # simple slice_minn
    exp = (penguins_data.nsmallest(3, 'bill_length_mm')
                        .sort_values('bill_length_mm')
                        .reset_index(drop = True)
                        )

    res = (penguins_tidy.slice_min(n = 3, order_by_column = 'bill_length_mm', with_ties = False)
                        .arrange('bill_length_mm')
                        .to_pandas()
                        )
    assert_frame_equal_v2(exp, res)

    # grouped slice_min
    exp = (penguins_data.groupby('species')
                        .apply(lambda x: x.nsmallest(3, 'bill_length_mm'))
                        .reset_index(drop = True)
                        .sort_values(['bill_length_mm', 'species'], ignore_index = True)
                        )

    res = (penguins_tidy.slice_min(n = 3,
                                   order_by_column = 'bill_length_mm',
                                   by = 'species',
                                   with_ties = False
                                   )
                        .arrange(['bill_length_mm', 'species'])
                        .to_pandas()
                        
                        )
                        
    assert_frame_equal_v2(exp, res)
    
    # test with_ties in slice_min/max
    exp = (penguins_tidy.slice_max(n = 3,
                                   order_by_column = 'year',
                                   by = 'species',
                                   with_ties = False
                                   )
                        )
    assert exp.nrow == 9
    
    exp = (penguins_tidy.slice_max(n = 3,
                                   order_by_column = 'year',
                                   by = 'species',
                                   with_ties = True
                                   )
                        )
    assert exp.nrow > 9
    
    
    exp = (penguins_tidy.slice_min(n = 3,
                                   order_by_column = 'year',
                                   by = 'species',
                                   with_ties = False
                                   )
                        )
    assert exp.nrow == 9
    
    exp = (penguins_tidy.slice_min(n = 3,
                                   order_by_column = 'year',
                                   by = 'species',
                                   with_ties = True
                                   )
                        )
    assert exp.nrow > 9


    # test for seed in grouped slice_sample
    exp1 = (penguins_tidy.slice_sample(n = 1, by = 'species', random_state = 100))
    exp2 = (penguins_tidy.slice_sample(n = 1, by = 'species', random_state = 100))
    
    assert_frame_equal_v2(exp1.to_pandas(), exp2.to_pandas())


# Test: expand_grid
def test_expand_grid(penguins_data):
    from tidypandas.tidy_utils import expand_grid
    import pandas as pd
    # use lists or series
    res_1 = expand_grid({'a': [1, 2], 'b': pd.Series(['m', 'n'])})
    assert isinstance(res_1, tidyframe)
    assert res_1.nrow == 4
    
    # dict value can be a tidyframe
    res_2 = expand_grid({'a': [1,2],
                         'b': tidyframe({'b': [3, pd.NA, 4], 'c': [5, 6, 7]})
                         })
    assert isinstance(res_2, tidyframe)
    assert res_2.nrow == 6

# Test: expand
def test_expand(penguins_data):
    penguins_tidy = tidyframe(load_penguins())

    # simple crossing
    res = penguins_tidy.expand(("species", "island"))
    assert isinstance(res, tidyframe)
    assert res.nrow == 9
    
    # simple nesting
    res = penguins_tidy.expand({"species", "island"})
    assert isinstance(res, tidyframe)
    assert res.nrow == 5
    
    # nest outside and crossing inside
    res = penguins_tidy.expand({"sex", ("species", "island")})
    assert isinstance(res, tidyframe)
    assert res.nrow == 13
    assert res.ncol == 3
    
    # crossing outside and nesting inside
    res = penguins_tidy.expand(("sex", {"species", "island"}))
    assert isinstance(res, tidyframe)
    assert res.nrow == 15
    assert res.ncol == 3
    
    # more 'nesting'
    res = penguins_tidy.expand((("year", "bill_length_mm"), {"species", "island"}))
    assert isinstance(res, tidyframe)
    assert res.nrow == 2475
    assert res.ncol == 4
    
    # grouped expand
    res = penguins_tidy.expand({"species", "island"}, by = 'sex')
    assert isinstance(res, tidyframe)
    assert res.nrow == 13
    assert res.ncol == 3
    
    # negative tests
    with pytest.raises(Exception) as e_info:
        # list input: tuple, set allowed
        penguins_tidy.expand(["species", "island"])
    
    with pytest.raises(Exception) as e_info:    
        # spec has length <= 1
        penguins_tidy.expand(set())
        penguins_tidy.expand(set('species'))
    
    with pytest.raises(Exception) as e_info:    
        # non-unique columns
        penguins_tidy.expand(("species", "island", {'species', 'year'}))
    
    with pytest.raises(Exception) as e_info:    
        # non-existing column
        penguins_tidy.expand({"species", "iceland"})
        
    with pytest.raises(Exception) as e_info:    
        # by should not intersect spec
        penguins_tidy.expand({"species", "iceland"}, by = "species")
        

# Test: complete
def test_complete(penguins_data):
    penguins_tidy = tidyframe(load_penguins())

    # simple crossing
    res = penguins_tidy.complete(("species", "island"))
    assert isinstance(res, tidyframe)
    assert res.nrow == 348
    assert res.ncol == 8
    
    # simple nesting
    res = penguins_tidy.complete({"species", "island"})
    assert isinstance(res, tidyframe)
    assert res.nrow == 344
    assert res.ncol == 8
    
    # nest outside and crossing inside
    res = penguins_tidy.complete({"sex", ("species", "island")})
    assert isinstance(res, tidyframe)
    assert res.nrow == 344
    assert res.ncol == 8
    
    # crossing outside and nesting inside
    res = penguins_tidy.complete(("sex", {"species", "island"}))
    assert isinstance(res, tidyframe)
    assert res.nrow == 346
    assert res.ncol == 8
    
    # more 'nesting'
    res = penguins_tidy.complete((("year", "bill_length_mm"), {"species", "island"}))
    assert isinstance(res, tidyframe)
    assert res.nrow == 2512
    assert res.ncol == 8
    
    # grouped expand
    res = penguins_tidy.complete({"species", "island"}, by = 'sex')
    assert isinstance(res, tidyframe)
    assert res.nrow == 344
    assert res.ncol == 8
    
    # negative tests
    # expand covers at its level
    # fill errors are handled by replace_na
    with pytest.raises(Exception) as e_info:    
        # by should not intersect spec
        penguins_tidy.complete({"species", "iceland"}, by = "species")
        
# Test: separate
def test_separate(penguins_data):
    tf = tidyframe(pd.DataFrame({'col': ["a_b", "c_d", "e_f_g"]}))
    res = tf.separate('col', sep = "_", into = ['A', 'B', 'C'], strict = False)
    assert isinstance(res, tidyframe)
    assert res.ncol == 3
    
    tf = tidyframe(pd.DataFrame({'col': ["a_b", pd.NA, "e_f_g"]}))
    tf.separate('col', sep = "_", into = ['A', 'B', 'C'], strict = False)
    assert isinstance(res, tidyframe)
    assert res.ncol == 3
    

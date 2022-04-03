import numpy as np
from palmerpenguins import load_penguins
from pandas._testing import assert_frame_equal
import pytest

from tidypandas import tidyframe, tidy
from tidypandas.tidy_utils import simplify

@pytest.fixture
def penguins_data():
    return simplify(load_penguins())


def test_mutate(penguins_data):

    penguins_tidy = tidy(penguins_data, copy=False)
    # Ungrouped
    # Type 1
    expected = penguins_data.assign(ind=np.arange(344))
    result = penguins_tidy.mutate({'ind' : np.arange(344)}).to_pandas(copy=False)
    assert_frame_equal(result, expected)

    # Type 2
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": lambda x: x['year'] + 1}).to_pandas()
    assert_frame_equal(result, expected)

    # Type 3
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": "x['year'] + 1"}).to_pandas()
    assert_frame_equal(result, expected)

    # Type 4
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": (lambda x: x + 1, "year")}).to_pandas()
    assert_frame_equal(result, expected)

    # Type 5
    expected = penguins_data.assign(next_year=lambda x: x['year'] + 1)
    result = penguins_tidy.mutate({"next_year": ("x + 1", "year")}).to_pandas()
    assert_frame_equal(result, expected)

    # Type 6
    expected = penguins_data.assign(x_plus_y=lambda x: x["bill_length_mm"]+x["bill_depth_mm"])
    result = penguins_tidy.mutate({"x_plus_y": (lambda x, y: x + y, ["bill_length_mm", "bill_depth_mm"])}).to_pandas()
    assert_frame_equal(result, expected)

    # Type 7 
    expected = penguins_data.assign(x=lambda x: x["year"]-1).assign(y=lambda x: x["year"]+1)
    result = penguins_tidy.mutate({('x', 'y'): lambda x: [x['year'] - 1, x['year'] + 1]}).to_pandas()
    assert_frame_equal(result, expected)

    # Type 8
    expected = penguins_data.assign(bill_length_mm=lambda x: x["bill_length_mm"]+1)
    result = penguins_tidy.mutate({'bill_length_mm': ("x + 1", )}).to_pandas()
    assert_frame_equal(result, expected)    

def test_mutate_by(penguins_data):
    penguins_tidy = tidy(penguins_data, copy=False)


    expected = penguins_data.groupby("sex", dropna=False)\
                  .apply(lambda x: x.assign(year_mod=lambda y: y["year"] + np.mean(y["year"]) - 4015))
    result = penguins_tidy.mutate({'year_mod' : "x['year'] + np.mean(x['year']) - 4015"}
                                   , by = 'sex'
                                   ).to_pandas()
    assert_frame_equal(expected, result)

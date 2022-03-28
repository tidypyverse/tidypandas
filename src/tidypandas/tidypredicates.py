## all predicates are implemented to return a closure which in turn takes in a TidyDataFrame
def starts_with(prefix):
    assert isinstance(prefix, str), "prefix in the predicate should be a string"
    # return [prefix == acol[0:min(len(prefix), len(acol))] for acol in cols]
    def func(df):
        cols = df.get_colnames()
        return [acol for acol in cols if prefix == acol[0:min(len(prefix), len(acol))]]
    return func


def ends_with(suffix):
    assert isinstance(suffix, str), "suffix in the predicate should be a string"
    # return [suffix == acol[max(-len(suffix), -len(acol)):] for acol in cols]
    def func(df):
        cols = df.get_colnames()
        return [acol for acol in cols if suffix == acol[max(-len(suffix), -len(acol)):]]
    return func


def everything():
    # return [True]*len(cols)
    def func(df):
        cols = df.get_colnames()
        return cols
    return func

def all_vars(cols, predicate):
    pass


def any_vars():
    pass

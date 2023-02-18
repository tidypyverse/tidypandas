def starts_with(prefix):
    assert(isinstance(prefix, str))
    def _starts_with_(tidy_df, prefix=prefix):
        cn = tidy_df.colnames
        sel_cn = list(filter(lambda x: x[0:len(prefix)] == prefix, cn))
        return sel_cn
    return _starts_with_


def ends_with(suffix):
    assert(isinstance(suffix, str))
    def _ends_with_(tidy_df, suffix=suffix):
        cn = tidy_df.colnames
        sel_cn = list(filter(lambda x: x[-len(suffix):] == suffix, cn))
        return sel_cn
    return _ends_with_

def contains(pattern):
    assert(isinstance(pattern, str))
    def _contains_(tidy_df, pattern=pattern):
        cn = tidy_df.colnames
        sel_cn = list(filter(lambda x: x.find(pattern) > -1, cn))
        return sel_cn
    return _contains_



        

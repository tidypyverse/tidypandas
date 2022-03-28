# -----------------------------------------------------------------------------
# This file is a part of tidypandas python package
# Find the dev version here: https://github.com/talegari/tidypandas
# -----------------------------------------------------------------------------

# import helpers
# import TidyDataFrame
# import TidyGroupedDataFrame 

def tidy(pdf):
    assert is_pdf(pdf)
    
    if not is_simple(pdf):
        pdf = simplify(pdf, verbose = True)
    
    if is_ungrouped_pdf(pdf):
        res = TidyDataFrame(pdf)
    else:
        res = TidyGroupedDataFrame(pdf)
        
    return res
    

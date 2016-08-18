import loading_target_data
import numpy as np

a = 4.5
b = " .4"

def test_int_erate():
    assert loading_target_data.int_erate(a) == 4.5
    assert loading_target_data.int_erate(b) == 0.4

c = np.NaN
d = "np.nan"
e = 'nan'
f = None
g = 'none'


def test_disqual():
    assert loading_target_data.disqual(c) == True
    assert loading_target_data.disqual(d) == False
    assert loading_target_data.disqual(e) == False
    assert loading_target_data.disqual(f) == True
    assert loading_target_data.disqual(g) == False


'''

this refers to int_erate in loading_target_data, and runs this local test_int_erate which,
 using the assert statements asks the function if a and b inputs correspond to the expected output of the function

'''

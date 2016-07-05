def none_func(series):
    f = series
    for ind, val in enumerate(f):
        f[ind] = get_rid_of_nones(val) 
        return(f)
        
        
def get_rid_of_nones(element):
    if element == None or '' or NoneType or np.NaN:
        return np.NaN
    elif element == "None":
        return np.NaN
    else:
        try:
            return str(float(element))
        except ValueError or TypeError:
            return np.NaN
            
results['salary'] = none_func(results['salary'])

#ISSUE IS THIS DOES NOT HANDLE NONES APPROPRIATELY 
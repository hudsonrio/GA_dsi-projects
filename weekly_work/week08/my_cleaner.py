def cleaner(lin):
    lin = lin.lower().replace('/',"").replace(" ", "").replace(".", "").replace("'", "").replace("?", "").replace("!", "").replace('"', "").replace(',', "")
    return(lin)

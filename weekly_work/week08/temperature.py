import pandas as pd
import numpy as np


def celsius_to_fahrenheit(temp):
  new_temp = (float(temp) * 9/5) +32
  return(new_temp)


def fahrenheit_to_celsius(temp):
  new_temp = (float(temp)-32) * 5/9
  return(new_temp)

  

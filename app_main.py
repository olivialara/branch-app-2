import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from app_functions import *

st.title('Laplace Mechanism')
st.subheader('Testing Varying Data Sizes, Ranges, Distributions, and Epsilon Values')
range_number = st.number_input('Select a size for the dataset:', min_value=0, max_value=1_000_000, value=10_000)
#st.write('Current size:', range_number)

lower = st.number_input('Select an approximate minimum value:', min_value=0, max_value=9_999, value=0)
upper = st.number_input('Select an approximate maximum value:', min_value=1, max_value=10_000, value=100)

# values = st.slider('Select a range of values:', 0, 10_000, (0, 100))
# st.write('Values:', values)

dist = st.radio('Which distribution would you like to use?',
     ('normal', 'uniform', 'poisson', 'binomial'))
#st.write('Distribution Type:', dist)

epsilon = st.number_input("Epsilon Value", min_value=0.0, max_value=10.0, value=0.5, step=0.1)

# df = pd.DataFrame(
#     np.random.randint(low=values[0], high = values[1], size = range_number, dtype=int),
#     columns=['true_values'])
# st.write(df)
# st.write( df.to_csv('specified_values.csv'))

new_df = st.write(create_dataset(dist, range_number, lower, upper, epsilon))
#st.write(new_df)
#st.write(new_df.describe())

#df_2 = st.write(add_noise_columns(new_df, epsilon, 1))
#st.write(df_2)





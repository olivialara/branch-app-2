import streamlit as st
import pandas as pd
import numpy as np
from statistics import mean
#plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

# numpy
import numpy as np
from numpy import random
from numpy.random import poisson

# sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# scipy
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import norm
from scipy.stats import kstest
from scipy.stats import chisquare
from scipy.special import rel_entr



def create_dataset(dist_type, data_size, lower, upper, epsilon):
    """
    This function generates a random dataset according to following parameters:
    dist_type: distribution type (normal, uniform, poisson, binomial)
    data_size: how large the dataset will be
    lower: the lowest possible value in dataset
    upper: the highest possible value in dataset
    """
    # create mean and std from lower and upper bounds
    mean = (lower + upper)/2
    std = (1/3) * mean
   
    if dist_type == "normal":
        # generate size random values according to normal distribution
        counts = list(np.round(np.random.normal(loc=mean, scale = std, size = data_size)))
        df = pd.DataFrame (counts, columns = ['counts'])
       
    elif dist_type == "uniform":
        # generate size random values according to uniform distribution
        counts = list(np.round(np.random.randint(low=0, high = upper + 1, size = data_size)))
        df = pd.DataFrame (counts, columns = ['counts'])
       
    elif dist_type == "poisson":
        # generate size random values according to poisson distribution
        counts = list(poisson(mean, data_size))
        df = pd.DataFrame(counts, columns = ['counts'])
       
    elif dist_type == "binomial":
        # generate size random values according to binomial distribution
        counts = list(random.binomial(n=data_size, p=mean/upper, size=data_size))
        df = pd.DataFrame(counts, columns = ['counts'])
       
    # return dataframe
    # print(f"Here is a randomly generated dataset according to a normal distribution of size {data_size} and with values ranging between {lower} and {upper}:")
    return(add_noise_columns(df, epsilon))

def add_noise_columns(df, epsilon):
    """
    This function adds various noises to the dataset according to following parameters:
    df: dataset that will have noise added to it
    epsilon: privacy budget (higher values result in less noise but more privacy leaked)
    sensitivity: The sensitivity of a function f is the amount f’s output changes when its input changes by 1
    """
    # add column that adds noise according to random laplace distribution
    df['laplace_noise_added'] = df['counts'] + [np.random.laplace(loc=0, scale = 1/epsilon) for idx in range(len(df))]
   
    # add column that rounds laplace noise
    df["laplace_noise_rounded"] = np.round(df["laplace_noise_added"])
   
    # add column that calculates the difference betweeen true and noisy values
    df['l_difference'] = df['counts'] - df['laplace_noise_added']
    # get values for min and max noise
    max_noise = df['l_difference'].max()
    min_noise = df['l_difference'].min()
    # get values for mean and std noise
    mean_noise = df['l_difference'].mean()
    std_noise = df['l_difference'].std()

    st.dataframe(df)
    return(graph_histograms(df))

# def kst_test(df):
#     p_val = kstest(df['counts'], df['laplace_noise_added'])[1]  
#     if p_val < 0.05:
#         caption = "the fit of the Laplace distribution is NOT good."
#     else:
#         caption = "the fit of the Laplace distribution IS good."
#     st.write(f'The P-value is {p_val}. Therefore, {caption}')
#     return(graph_histograms(df))

def graph_histograms(df):
    # HISTOGRAMS
    # overlayed histograms to see varying distributions
    # fig = go.Figure()
    # fig.add_trace(go.Histogram(x=df['counts'], histnorm = 'probability', marker_color = '#0000FF', name = 'True Counts'))
    # fig.add_trace(go.Histogram(x=df['laplace_noise_added'], histnorm = 'probability', marker_color = '#6ed85f', name= 'Laplace Noise Added'))
    # fig.add_trace(go.Histogram(x=df['uniform_noise_added'], histnorm = 'probability', marker_color = '#4b90da', name = 'Uniform Noise Added'))
    # fig.add_trace(go.Histogram(x=df['normal_noise_added'], histnorm = 'probability', marker_color = 'orange', name = 'Normal Noise Added'))

    # overlay both histograms and add titles
    # fig.update_layout(barmode='overlay', xaxis_title_text='Counts', xaxis label, yaxis_title_text='Probability')
    # fig.update_layout(title = '<b>Histograms of All Noises</b>', title_x = .5)

    # reduce opacity to see both histograms
    # fig.update_traces(opacity=0.5)
    # print(fig.show())
   
    #HISTOGRAMS (JUST TRUE AND LAPLACE)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['counts'], histnorm = 'probability', marker_color = '#0000FF', name = 'True Counts'))
    fig.add_trace(go.Histogram(x=df['laplace_noise_added'], histnorm = 'probability', marker_color = '#6ed85f', name= 'Laplace Noise Added'))

    # overlay both histograms and add titles
    fig.update_layout(barmode='overlay',
                      xaxis_title_text='Counts', # xaxis label
                      yaxis_title_text='Frequency',)

    fig.update_layout(title = '<b>Histograms of True vs Laplace Noise Added</b>', title_x = .5)

    # reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    return(fig)
   
    # # CURVES
    # #hist_data = [df['counts'], df['laplace_noise_added'], df['uniform_noise_added'], df['normal_noise_added']]
    # hist_data = [df['counts'], df['laplace_noise_added']]

    # #group_labels = ['True Counts', 'Laplace Noise Added', 'Uniform Noise Added', 'Normal Noise Added']
    # group_labels = ['True Counts', 'Laplace Noise Added']
    # #colors = ['#0000FF','#6ed85f', '#4b90da', 'orange']
    # colors = ['#0000FF','#6ed85f']

    # # Create distplot with curve_type set to 'normal'
    # fig = ff.create_distplot(hist_data, group_labels, colors=colors,
    #                          bin_size=.2, show_rug=False, show_hist = False)

    # # Add title
    # fig.update_layout(title_text='<b>Density Curves of True vs Laplace Noise Added</b>', title_x = .5,
    #                   xaxis_title_text='Counts',
    #                   yaxis_title_text='Probability')
    # #print(fig.show())
    # return(fig)



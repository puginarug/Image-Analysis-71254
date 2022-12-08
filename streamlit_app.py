# import libs
import streamlit as st
import cv2
import numpy as np
import skimage.io as io
#import matplotlib.pyplot as plt

# check versions
#np.__version__

# main page
st.set_page_config(page_title='David Cookie Finder', layout='wide',
                   initial_sidebar_state='auto')
st.title('Cookie Finder using K-Means, by David Zingerman')

# side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }

    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>

    """,
    unsafe_allow_html=True,

)

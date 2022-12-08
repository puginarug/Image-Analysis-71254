# import libs
import streamlit as st
import cv2
import numpy as np
import skimage.io as io
#import matplotlib.pyplot as plt

# check versions
#np.__version__

# function to segment using k-means

def segment_image_kmeans(img, k=3, attempts=10):
    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values = img.reshape((-1, 3))  # -1 reshape means, in this case MxN

    # We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)

    return segmented_image


# vars
DEMO_IMAGE = 'demo.png'  # a demo image for the segmentation page, if none is uploaded
favicon = 'favicon.png'

# main page
st.set_page_config(page_title='K-Means - Yedidya Harris', page_icon=favicon, layout='wide',
                   initial_sidebar_state='auto')
st.title('Image Segmentation using K-Means, by Yedidya Harris')

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

st.sidebar.title('Segmentation Sidebar')
st.sidebar.subheader('Site Pages')


# using st.cache so streamlit runs the following function only once, and stores in chache (until changed)
@st.cache()
# take an image, and return a resized that fits our page
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

# add dropdown to select pages on left
app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'Segment an Image'])

# About page
if app_mode == 'About App':
    st.markdown('In this app we will segment images using K-Means')

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

    # add a video to the page
    st.video('https://www.youtube.com/watch?v=6CqRnx6Ic48')

    st.markdown('''
                ## About the app \n
                Hey, this web app is a great one to segment images using K-Means. \n
                There are many way. \n
                Enjoy! Yedidya


                ''')

    # Run image
    if app_mode == 'Segment an Image':

        st.sidebar.markdown('---')  # adds a devider (a line)

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

        # choosing a k value (either with +- or with a slider)
        k_value = st.sidebar.number_input('Insert K value (number of clusters):', value=4,
                                          min_value=1)  # asks for input from the user
        st.sidebar.markdown('---')  # adds a devider (a line)

        attempts_value_slider = st.sidebar.slider('Number of attempts', value=7, min_value=1,
                                                  max_value=10)  # slider example
        st.sidebar.markdown('---')  # adds a devider (a line)

        # read an image from the user
        img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

        # assign the uplodaed image from the buffer, by reading it in
        if img_file_buffer is not None:
            image = io.imread(img_file_buffer)
        else:  # if no image was uploaded, then segment the demo image
            demo_image = DEMO_IMAGE
            image = io.imread(demo_image)

        # display on the sidebar the uploaded image
        st.sidebar.text('Original Image')
        st.sidebar.image(image)

        # call the function to segment the image
        segmented_image = segment_image_kmeans(image, k=k_value, attempts=attempts_value_slider)

        # Display the result on the right (main frame)
        st.subheader('Output Image')
        st.image(segmented_image, use_column_width=True)
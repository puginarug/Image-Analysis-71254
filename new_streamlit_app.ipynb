{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3d733a8e",
   "metadata": {
    "executionInfo": {
     "elapsed": 2492,
     "status": "ok",
     "timestamp": 1670514645782,
     "user": {
      "displayName": "David Zingerman",
      "userId": "12286570890350507890"
     },
     "user_tz": -120
    },
    "id": "3d733a8e",
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# import libs\n",
    "import streamlit as st\n",
    "import cv2\n",
    "import numpy as np\n",
    "import skimage.io as io\n",
    "#import matplotlib.pyplot as plt\n",
    "\n",
    "# check versions\n",
    "#np.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c37e0a4",
   "metadata": {
    "id": "9c37e0a4"
   },
   "outputs": [],
   "source": [
    "# function to segment using k-means\n",
    "\n",
    "def segment_image_kmeans(img, k=3, attempts=10): \n",
    "\n",
    "    # Convert MxNx3 image into Kx3 where K=MxN\n",
    "    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN\n",
    "\n",
    "    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV\n",
    "    pixel_values = np.float32(pixel_values)\n",
    "\n",
    "    # define stopping criteria\n",
    "    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)\n",
    "    \n",
    "    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)\n",
    "    \n",
    "    # convert back to 8 bit values\n",
    "    centers = np.uint8(centers)\n",
    "\n",
    "    # flatten the labels array\n",
    "    labels = labels.flatten()\n",
    "    \n",
    "    # convert all pixels to the color of the centroids\n",
    "    segmented_image = centers[labels.flatten()]\n",
    "    \n",
    "    # reshape back to the original image dimension\n",
    "    segmented_image = segmented_image.reshape(img.shape)\n",
    "    \n",
    "    return segmented_image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "735964f5",
   "metadata": {
    "id": "735964f5",
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# vars\n",
    "DEMO_IMAGE = 'demo.png' # a demo image for the segmentation page, if none is uploaded\n",
    "favicon = 'favicon.png'\n",
    "\n",
    "# main page\n",
    "st.set_page_config(page_title='K-Means - Yedidya Harris', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')\n",
    "st.title('Image Segmentation using K-Means, by Yedidya Harris')\n",
    "\n",
    "# side bar\n",
    "st.markdown(\n",
    "    \"\"\"\n",
    "    <style>\n",
    "    [data-testid=\"stSidebar\"][aria-expanded=\"true\"] . div:first-child{\n",
    "        width: 350px\n",
    "    }\n",
    "    \n",
    "    [data-testid=\"stSidebar\"][aria-expanded=\"false\"] . div:first-child{\n",
    "        width: 350px\n",
    "        margin-left: -350px\n",
    "    }    \n",
    "    </style>\n",
    "    \n",
    "    \"\"\",\n",
    "    unsafe_allow_html=True,\n",
    "\n",
    "\n",
    ")\n",
    "\n",
    "st.sidebar.title('Segmentation Sidebar')\n",
    "st.sidebar.subheader('Site Pages')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "967ee042",
   "metadata": {
    "id": "967ee042",
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# using st.cache so streamlit runs the following function only once, and stores in chache (until changed)\n",
    "@st.cache()\n",
    "\n",
    "# take an image, and return a resized that fits our page\n",
    "def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):\n",
    "    dim = None\n",
    "    (h,w) = image.shape[:2]\n",
    "    \n",
    "    if width is None and height is None:\n",
    "        return image\n",
    "    \n",
    "    if width is None:\n",
    "        r = width/float(w)\n",
    "        dim = (int(w*r),height)\n",
    "    \n",
    "    else:\n",
    "        r = width/float(w)\n",
    "        dim = (width, int(h*r))\n",
    "        \n",
    "    # resize the image\n",
    "    resized = cv2.resize(image, dim, interpolation=inter)\n",
    "    \n",
    "    return resized"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8e73676",
   "metadata": {
    "id": "d8e73676",
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# add dropdown to select pages on left\n",
    "app_mode = st.sidebar.selectbox('Navigate',\n",
    "                                  ['About App', 'Segment an Image'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0490187c",
   "metadata": {
    "id": "0490187c",
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# About page\n",
    "if app_mode == 'About App':\n",
    "    st.markdown('In this app we will segment images using K-Means')\n",
    "    \n",
    "    \n",
    "    # side bar\n",
    "    st.markdown(\n",
    "        \"\"\"\n",
    "        <style>\n",
    "        [data-testid=\"stSidebar\"][aria-expanded=\"true\"] . div:first-child{\n",
    "            width: 350px\n",
    "        }\n",
    "\n",
    "        [data-testid=\"stSidebar\"][aria-expanded=\"false\"] . div:first-child{\n",
    "            width: 350px\n",
    "            margin-left: -350px\n",
    "        }    \n",
    "        </style>\n",
    "\n",
    "        \"\"\",\n",
    "        unsafe_allow_html=True,\n",
    "\n",
    "\n",
    "    )\n",
    "\n",
    "    # add a video to the page\n",
    "    st.video('https://www.youtube.com/watch?v=6CqRnx6Ic48')\n",
    "\n",
    "\n",
    "    st.markdown('''\n",
    "                ## About the app \\n\n",
    "                Hey, this web app is a great one to segment images using K-Means. \\n\n",
    "                There are many way. \\n\n",
    "                Enjoy! Yedidya\n",
    "\n",
    "\n",
    "                ''') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6f539e86",
   "metadata": {
    "id": "6f539e86",
    "lines_to_next_cell": 3
   },
   "outputs": [],
   "source": [
    "# Run image\n",
    "if app_mode == 'Segment an Image':\n",
    "    \n",
    "    st.sidebar.markdown('---') # adds a devider (a line)\n",
    "    \n",
    "    # side bar\n",
    "    st.markdown(\n",
    "        \"\"\"\n",
    "        <style>\n",
    "        [data-testid=\"stSidebar\"][aria-expanded=\"true\"] . div:first-child{\n",
    "            width: 350px\n",
    "        }\n",
    "\n",
    "        [data-testid=\"stSidebar\"][aria-expanded=\"false\"] . div:first-child{\n",
    "            width: 350px\n",
    "            margin-left: -350px\n",
    "        }    \n",
    "        </style>\n",
    "\n",
    "        \"\"\",\n",
    "        unsafe_allow_html=True,\n",
    "\n",
    "\n",
    "    )\n",
    "\n",
    "    # choosing a k value (either with +- or with a slider)\n",
    "    k_value = st.sidebar.number_input('Insert K value (number of clusters):', value=4, min_value = 1) # asks for input from the user\n",
    "    st.sidebar.markdown('---') # adds a devider (a line)\n",
    "    \n",
    "    attempts_value_slider = st.sidebar.slider('Number of attempts', value = 7, min_value = 1, max_value = 10) # slider example\n",
    "    st.sidebar.markdown('---') # adds a devider (a line)\n",
    "    \n",
    "    # read an image from the user\n",
    "    img_file_buffer = st.sidebar.file_uploader(\"Upload an image\", type=['jpg', 'jpeg', 'png'])\n",
    "\n",
    "    # assign the uplodaed image from the buffer, by reading it in\n",
    "    if img_file_buffer is not None:\n",
    "        image = io.imread(img_file_buffer)\n",
    "    else: # if no image was uploaded, then segment the demo image\n",
    "        demo_image = DEMO_IMAGE\n",
    "        image = io.imread(demo_image)\n",
    "\n",
    "    # display on the sidebar the uploaded image\n",
    "    st.sidebar.text('Original Image')\n",
    "    st.sidebar.image(image)\n",
    "    \n",
    "    # call the function to segment the image\n",
    "    segmented_image = segment_image_kmeans(image, k=k_value, attempts=attempts_value_slider)\n",
    "    \n",
    "    # Display the result on the right (main frame)\n",
    "    st.subheader('Output Image')\n",
    "    st.image(segmented_image, use_column_width=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d4oBGNJi8rhE",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 18329,
     "status": "ok",
     "timestamp": 1670515396293,
     "user": {
      "displayName": "David Zingerman",
      "userId": "12286570890350507890"
     },
     "user_tz": -120
    },
    "id": "d4oBGNJi8rhE",
    "outputId": "a53910f7-7d3f-4b93-966c-6304a258d8c0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mounted at /content/drive\n"
     ]
    }
   ],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "vSXLa5cD-qxH",
   "metadata": {
    "id": "vSXLa5cD-qxH"
   },
   "outputs": [],
   "source": [
    "# first remove the markdown cells (because it causes problems in streamlit app)\n",
    "\n",
    "import nbformat as nbf\n",
    "folder_path =r'/content/drive/MyDrive/71254_2023/02_Assignment_Submission/David.Zingerman/Class_Exercise'\n",
    "ntbk_name_to_convert = 'streamlit_app.ipynb'\n",
    "ntbk = nbf.read(f'{folder_path}/kmeans.ipynb', nbf.NO_CONVERT)\n",
    "new_ntbk = ntbk\n",
    "new_ntbk.cells = [cell for cell in ntbk.cells if cell.cell_type != \"markdown\"]\n",
    "nbf.write(new_ntbk, f'new_{ntbk_name_to_convert}', version=nbf.NO_CONVERT)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "QMit_Ozq_LCQ",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8440,
     "status": "ok",
     "timestamp": 1670507578036,
     "user": {
      "displayName": "Yedidya Harris",
      "userId": "06328729226508388184"
     },
     "user_tz": -120
    },
    "id": "QMit_Ozq_LCQ",
    "outputId": "c375e538-868e-4950-aa7a-b59bdf9988d4"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
      "Requirement already satisfied: ipython in /usr/local/lib/python3.8/dist-packages (7.9.0)\n",
      "Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.8/dist-packages (from ipython) (5.6.0)\n",
      "Requirement already satisfied: pexpect in /usr/local/lib/python3.8/dist-packages (from ipython) (4.8.0)\n",
      "Requirement already satisfied: decorator in /usr/local/lib/python3.8/dist-packages (from ipython) (4.4.2)\n",
      "Requirement already satisfied: prompt-toolkit<2.1.0,>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from ipython) (2.0.10)\n",
      "Requirement already satisfied: jedi>=0.10 in /usr/local/lib/python3.8/dist-packages (from ipython) (0.18.2)\n",
      "Requirement already satisfied: pygments in /usr/local/lib/python3.8/dist-packages (from ipython) (2.6.1)\n",
      "Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.8/dist-packages (from ipython) (57.4.0)\n",
      "Requirement already satisfied: pickleshare in /usr/local/lib/python3.8/dist-packages (from ipython) (0.7.5)\n",
      "Requirement already satisfied: backcall in /usr/local/lib/python3.8/dist-packages (from ipython) (0.2.0)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.0 in /usr/local/lib/python3.8/dist-packages (from jedi>=0.10->ipython) (0.8.3)\n",
      "Requirement already satisfied: wcwidth in /usr/local/lib/python3.8/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython) (0.2.5)\n",
      "Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.8/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython) (1.15.0)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.8/dist-packages (from pexpect->ipython) (0.7.0)\n",
      "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
      "Requirement already satisfied: nbconvert in /usr/local/lib/python3.8/dist-packages (5.6.1)\n",
      "Requirement already satisfied: testpath in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.6.0)\n",
      "Requirement already satisfied: jupyter-core in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.1.0)\n",
      "Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.6.0)\n",
      "Requirement already satisfied: pygments in /usr/local/lib/python3.8/dist-packages (from nbconvert) (2.6.1)\n",
      "Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.4)\n",
      "Requirement already satisfied: bleach in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.0.1)\n",
      "Requirement already satisfied: defusedxml in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.7.1)\n",
      "Requirement already satisfied: jinja2>=2.4 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (2.11.3)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (1.5.0)\n",
      "Requirement already satisfied: nbformat>=4.4 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.7.0)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.8.4)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.8/dist-packages (from jinja2>=2.4->nbconvert) (2.0.1)\n",
      "Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.8/dist-packages (from nbformat>=4.4->nbconvert) (4.3.3)\n",
      "Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.8/dist-packages (from nbformat>=4.4->nbconvert) (2.16.2)\n",
      "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (22.1.0)\n",
      "Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (5.10.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (0.19.2)\n",
      "Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.8/dist-packages (from importlib-resources>=1.4.0->jsonschema>=2.6->nbformat>=4.4->nbconvert) (3.11.0)\n",
      "Requirement already satisfied: webencodings in /usr/local/lib/python3.8/dist-packages (from bleach->nbconvert) (0.5.1)\n",
      "Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.8/dist-packages (from bleach->nbconvert) (1.15.0)\n",
      "Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.8/dist-packages (from jupyter-core->nbconvert) (2.5.4)\n",
      "[TerminalIPythonApp] WARNING | Subcommand `ipython nbconvert` is deprecated and will be removed in future versions.\n",
      "[TerminalIPythonApp] WARNING | You likely want to use `jupyter nbconvert` in the future\n",
      "[NbConvertApp] Converting notebook new_kmeans.ipynb to python\n",
      "[NbConvertApp] Writing 6662 bytes to new_kmeans.py\n"
     ]
    }
   ],
   "source": [
    "# then convert to .py format\n",
    "!pip install ipython\n",
    "!pip install nbconvert\n",
    "\n",
    "# the conversion (it'll save the .py file on the left, in the content)\n",
    "!ipython nbconvert new_kmeans.ipynb --to python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "j5tQux9bIJ9z",
   "metadata": {
    "id": "j5tQux9bIJ9z"
   },
   "outputs": [],
   "source": [
    "# add more libraries if you used! as a new line\n",
    "with open('requirements.txt', 'w') as f:\n",
    "    f.write('''streamlit\n",
    "scikit-image\n",
    "opencv-contrib-python-headless\n",
    "numpy''')"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "bT6Nqvz-K-ax"
   ],
   "provenance": [
    {
     "file_id": "17Fk8zectxRB39ueE9IortqaHGvbrtr8C",
     "timestamp": 1670513786833
    }
   ]
  },
  "gpuClass": "standard",
  "jupytext": {
   "cell_metadata_filter": "-all",
   "encoding": "# coding: utf-8",
   "executable": "/usr/bin/env python",
   "main_language": "python",
   "notebook_metadata_filter": "-all"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

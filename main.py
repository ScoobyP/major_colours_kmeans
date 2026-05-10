import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import streamlit as st

st.title("Clustering Major Colours from Images")
st.header("Please upload an image")
# Uploading the image
uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "avif", "heic", "heif", "webp"]
)



# Displaying the image
if uploaded_file is not None:
    img = plt.imread(uploaded_file)
    plt.imshow(img)
    plt.axis("off")
    st.image(img)
    st.write(img.shape)
    len, br, rgb = img.shape



    if st.button("Process Image", use_container_width=True):
        #Processing image
        arrayed_img = np.array(img).reshape(-1, 3)

        #Applying kmeans
        mini_batch_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=4000, init="k-means++")
        mini_batch_kmeans.fit(arrayed_img)

        labels = mini_batch_kmeans.labels_
        centers = mini_batch_kmeans.cluster_centers_

        compressed_img = np.array(centers[labels], dtype = "uint8").reshape(len, br, rgb)
        plt.imshow(compressed_img)
        st.image(compressed_img)

        #converting labels into series and then getting value_counts() index
        labels_df = pd.Series(labels)
        dominant_indices = labels_df.value_counts(ascending=False).index

        #masking those indices to the cluster_centers to get colours
        colours = np.array(centers[dominant_indices], dtype = "uint8")
        st.header("Dominant Colours")

        palette = colours.reshape(1, -1, 3)
        palette = np.repeat(palette, 350, axis=0)
        palette = np.repeat(palette, 750, axis=1)

        st.image(palette)




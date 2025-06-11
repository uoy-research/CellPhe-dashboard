import time
import os
from pathlib import Path
import shutil
import warnings

from cellphe import (
    track_images,
    cell_features,
    import_data,
    time_series_features,
)
import cellphe
from cellpose import models
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from skimage import io
import streamlit as st
import umap

# Suppress warnings
warnings.filterwarnings("ignore")

EXCLUDE_ANALYSIS_COLUMNS = ["CellID", "FrameID", "ROI_filename"]
ARCHIVE_FN = "outputs.zip"


def segment_images_with_progress_bar(
    image_folder,
    masks_folder,
    model_params={'model_type': 'cyto3'},
    eval_params = {}
):
    """
    Local version of cellphe's `segment_images`.
    Does the same job but with 2 modifications:
         - Uses CellposeModel rather than Cellpose, which allows for custom
             cellpose models
         - Updates a progress bar after each image is segmented
    """
    model = models.CellposeModel(**model_params)
    image_fns = sorted(os.listdir(image_folder))
    n_images = len(image_fns)
    segmenting_bar = st.progress(0, text=f"Image {0}/{n_images}")
    for i, fn in enumerate(image_fns):
        image = np.array(Image.open(os.path.join(image_folder, fn)))
        masks = model.eval(image, **eval_params)[0]
        segmenting_bar.progress((i+1)/n_images, text=f"Image {i+1}/{n_images}")
        io.imsave(os.path.join(masks_folder, fn), masks.astype("uint16"))  # Assuming masks are uint16
    time.sleep(2)
    segmenting_bar.empty()


# Function to assign feature categories based on substrings in the feature names
def assign_color(feature, colour_mapping):
    # Check for substrings to assign to a group
    if (
        "Vol" in feature
        or "Rad" in feature
        or "Wid" in feature
        or "Len" in feature
        or "Area" in feature
    ):
        return colour_mapping["size"]
    elif (
        "Sph" in feature
        or "Box" in feature
        or "Rect" in feature
        or "VfC" in feature
        or "Cur" in feature
        or "A2B" in feature
        or "Poly" in feature
    ):
        return colour_mapping["shape"]
    elif "FO" in feature or "Cooc" in feature or "IQ" in feature:
        return colour_mapping["texture"]
    elif (
        feature == "x"
        or feature == "y"
        or "trajArea" in feature
        or "Vel" in feature
        or "Dis" in feature
        or "D2T" in feature
        or "Trac" in feature
    ):
        return colour_mapping["movement"]
    else:
        return colour_mapping["density"]


def process_images(
    raw_images,
    framerate=0.0028,
    min_frames=0,
    keep_masks=False,
    keep_rois=False,
    keep_trackmate_features=False,
    keep_cellphe_frame_features=False,
    cellpose_model='cyto3',
    uploaded_masks=None,
    uploaded_roi=None,
    uploaded_trackmate_csv=None
):
    """
    Process a folder of images to extract cell features and time series features.

    Parameters:
    - raw_images: list[str], List of uploaded image paths
    - framerate: float, frame rate for time series analysis (default: 0.0028)

    Returns:
    - tsvariables: DataFrame, extracted time series features (if applicable)
    """
    # Step 1: Define paths
    out_dir = "outputs"
    image_folder = "images"
    masks_folder = os.path.join(out_dir, "masks")
    rois_archive = os.path.join(out_dir, "rois.zip")
    trackmate_csv = os.path.join(out_dir, "trackmate.csv")
    frame_features_csv = os.path.join(out_dir, "frame_features.csv")
    ts_features_csv = os.path.join(out_dir, "time_series_features.csv")
    Path(masks_folder).mkdir(parents=True, exist_ok=True)
    Path(image_folder).mkdir(exist_ok=True)

    # Write uploaded files to disk - unnecesary file IO but CellPhe requires
    # images to be on disk, not in memory
    for file in raw_images:
        with open(os.path.join(image_folder, file.name), 'wb') as outfile:
            outfile.write(file.getvalue())

    if len(uploaded_masks) > 0:
        for file in uploaded_masks:
            with open(os.path.join(masks_folder, file.name), 'wb') as outfile:
                outfile.write(file.getvalue())

    if uploaded_roi is not None:
        with open(rois_archive, 'wb') as outfile:
            outfile.write(uploaded_roi.getvalue())

    if uploaded_trackmate_csv is not None:
        with open(trackmate_csv, 'wb') as outfile:
            outfile.write(uploaded_trackmate_csv.getvalue())

    have_masks = len(uploaded_masks) > 0
    have_tracking = uploaded_roi is not None and uploaded_trackmate_csv is not None


    # Step 2: Segment images
    overall_bar = st.progress(0.2, text="Segmenting")
    if not have_masks and not have_tracking:
        try:
            if cellpose_model == 'cyto3':
                cellpose_params = {'model_type': cellpose_model}
            else:
                if cellpose_model == 'ioLight':
                    model_path = "cellpose_models/CP_20250421_ioLight_21imgs"
                elif cellpose_model == 'LiveCyte Brightfield':
                    model_path = "cellpose_models/CP_20250502_Livcyto_25imgs"
                else:
                    model_path = ''  # appease linter, code can't get here
                cellpose_params = {'pretrained_model': model_path}
            segment_images_with_progress_bar(
                image_folder,
                masks_folder,
                model_params=cellpose_params
            )
        except Exception as e:
            st.write(f"An unexpected error occurred during segmentation: {e}")
            overall_bar.empty()
            return

    # Step 3: Track images
    overall_bar.progress(0.4, text="Tracking")
    if not have_tracking:
        try:
            track_images(masks_folder, trackmate_csv, rois_archive)
        except Exception as e:
            st.write(f"An unexpected error occurred during tracking: {e}")
            overall_bar.empty()
            return

    # Step 4: Import tracked data
    try:
        feature_table = import_data(trackmate_csv, "Trackmate_auto", min_frames)
    except:
        st.write("Unable to import tracked data")
        overall_bar.empty()
        return

    if feature_table.shape[0] == 0:
        st.write("No cells found")
        overall_bar.empty()
        return

    # Step 5: Extract features
    overall_bar.progress(0.6, text="Extracting frame features")
    try:
        frame_features = cell_features(
            feature_table, rois_archive, image_folder, framerate=framerate
        )
    except:
        st.write("An error occured while extracting the frame level features")
        overall_bar.empty()
        return

    # Step 6: Extract time series features
    overall_bar.progress(0.8, text="Extracting temporal features")
    # Extract time series features
    try:
        tsvariables = time_series_features(frame_features)
    except:
        st.write("An error occured while extracting the temporal features, NB: generally at least 20 frames are needed for robust calculation")
        overall_bar.empty()
        return

    # Save the new features to a CSV file
    tsvariables.to_csv(ts_features_csv, index=False)
    overall_bar.progress(1, text="Processing complete")
    time.sleep(2)

    overall_bar.empty()

    if not keep_masks:
        shutil.rmtree(masks_folder)
    if not keep_rois:
        os.remove(rois_archive)
    if not keep_trackmate_features:
        os.remove(trackmate_csv)
    if keep_cellphe_frame_features:
        frame_features.to_csv(frame_features_csv, index=False)

    shutil.make_archive("outputs", "zip", "outputs")

    return tsvariables


# Function for plotting separation scores and PCA side by side
def plot_sep_and_pca_side_by_side(sep_df, top_features, data, labels):
    """
    Plots the separation scores and PCA plots side by side using Streamlit columns.

    Parameters:
    - sep_df: DataFrame containing separation scores
    - top_features: List of top discriminatory features
    - data: DataFrame containing the feature data for PCA
    - labels: List of group labels corresponding to the data
    """

    colour_mapping = {
        "size": "black",  # For size-related features
        "shape": "mediumturquoise",  # For shape-related features
        "texture": "mediumpurple",  # For texture-related features (example)
        "movement": "cornflowerblue",  # For movement-related features
        "density": "hotpink",  # For density-related features
    }

    # Apply color assignment to the DataFrame based on the feature names
    sep_df["Colour"] = sep_df["Feature"].apply(
        assign_color, colour_mapping=colour_mapping
    )

    # Create two columns for side-by-side plotting
    col1, col2 = st.columns(2, vertical_alignment="center")
    col3, col4 = st.columns(2, vertical_alignment="center")

    # Column 1: Separation Scores
    with col1:
        plt.figure(figsize=(6, 6))  # Set equal size for the separation scores plot
        sns.barplot(
            data=sep_df, x="Feature", y="Separation", palette=sep_df["Colour"].tolist()
        )  # Convert to list
        plt.title("Separation Scores for Each Feature")
        plt.xlabel("Feature Names")
        plt.ylabel("Separation Score")
        plt.xticks(rotation=90)  # Rotate feature names for better visibility

        # Creating a legend
        handles = []
        for label, color in colour_mapping.items():
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=color,
                    markersize=10,
                )
            )
        plt.legend(title="Feature Categories", handles=handles)

        st.pyplot(plt.gcf())

    # Preprocess data for dimensionality reduction
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[top_features])

    # Create a DataFrame with the scaled data for easier handling
    scaled_df = pd.DataFrame(scaled_data, columns=top_features)
    scaled_df["Group"] = labels

    # Drop rows with NaN values before PCA
    scaled_df = scaled_df.dropna()

    # Adjust labels to match the remaining rows after dropping NaNs
    labels_cleaned = scaled_df["Group"].values
    scaled_data_cleaned = scaled_df[top_features].values

    # Column 2: PCA Plot
    with col2:
        # Perform PCA on the cleaned, scaled data
        pca = PCA(n_components=2)  # Use 2 principal components for visualization
        pca_scores = pca.fit_transform(scaled_data_cleaned)

        # Create a DataFrame for PCA scores
        pca_df = pd.DataFrame(pca_scores, columns=["PC1", "PC2"])
        pca_df["Group"] = labels_cleaned

        # Plot PCA scores
        plt.figure(figsize=(6, 6))  # Set equal size for the PCA plot
        sns.scatterplot(
            data=pca_df, x="PC1", y="PC2", hue="Group", palette="tab10", s=100
        )

        plt.title("PCA")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)")
        plt.legend(title="Group")
        st.pyplot(plt.gcf())

    # Column 3: tSNE Plot
    with col3:
        # Perform tSNE on the cleaned, scaled data
        tsne_mod = TSNE(init="random", perplexity=3)
        tsne_scores = tsne_mod.fit_transform(scaled_data_cleaned)

        # Create a DataFrame for tSNE scores
        tsne_df = pd.DataFrame(tsne_scores, columns=["Dim1", "Dim2"])
        tsne_df["Group"] = labels_cleaned

        # Plot tSNE scores
        plt.figure(figsize=(6, 6))  # Set equal size for the PCA plot
        sns.scatterplot(
            data=tsne_df, x="Dim1", y="Dim2", hue="Group", palette="tab10", s=100
        )

        plt.title("tSNE")
        plt.xlabel("")
        plt.ylabel("")
        plt.legend(title="Group")
        st.pyplot(plt.gcf())

    # Column 4: UMAP Plot
    with col4:
        # Perform UMAP on the cleaned, scaled data
        umap_mod = umap.UMAP()
        umap_scores = umap_mod.fit_transform(scaled_data_cleaned)

        # Create a DataFrame for UMAP scores
        umap_df = pd.DataFrame(umap_scores, columns=["Dim1", "Dim2"])
        umap_df["Group"] = labels_cleaned

        # Plot UMAP scores
        plt.figure(figsize=(6, 6))  # Set equal size for the UMAP plot
        sns.scatterplot(
            data=umap_df, x="Dim1", y="Dim2", hue="Group", palette="tab10", s=100
        )

        plt.title("UMAP")
        plt.xlabel("")
        plt.ylabel("")
        plt.legend(title="Group")
        st.pyplot(plt.gcf())


# Function for plotting boxplots
def plot_boxplot_with_points(data, feature, labels):
    """
    Plots box plots with individual data points for the specified feature,
    where points are slightly darker and larger than the box colors.

    Parameters:
    - data: DataFrame containing cell features
    - feature: str, the feature to plot
    - labels: list, group labels for the data
    """
    plt.figure(figsize=(12, 6))

    # Create a new DataFrame for plotting
    plot_data = pd.DataFrame({"Feature": data[feature], "Group": labels})

    # Get color palette from 'tab10'
    palette = sns.color_palette("tab10")

    # Create the box plot without transparency first
    boxplot = sns.boxplot(
        x="Group",
        y="Feature",
        data=plot_data,
        palette=palette,
        showfliers=False,
        width=0.5,
    )

    # Set the transparency for the boxes
    for patch in boxplot.artists:
        patch.set_alpha(0.1)  # Set transparency for the boxes

    # Adjust the color palette to be slightly darker for the points
    darker_palette = [mcolors.to_rgba(c, alpha=1) for c in palette]

    # Overlay individual data points, slightly darker and larger than the boxes
    sns.stripplot(
        x="Group",
        y="Feature",
        data=plot_data,
        palette=darker_palette,
        alpha=0.8,
        size=8,
    )  # Increase point size and darken

    plt.title(f"Box Plot for {feature}")
    plt.xlabel("Group")
    plt.ylabel(feature)

    # Display the plot
    st.pyplot(plt.gcf())


# Streamlit app structure
st.title("The CellPhe Toolkit for Cell Phenotyping")

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["Image Processing", "Single Population", "Multiple Populations"]
)

# Tab 1: Image Processing
with tab1:
    st.markdown("# Image Processing")
    st.markdown(
        """
                Run the CellPhe pipeline from start to finish on your images.
                This contains several steps and will take some time.
                  1. Segments the images using CellPose
                  2. Tracks the images using TrackMate
                  3. Imports the tracked and segmented images into CellPhe
                  4. Generates the CellPhe cell features for each frame
                  5. Generate the CellPhe summary features for each cell across
                the entire time-lapse.

                NB: the time-series summary features work best on cells tracked
                for a long time. By default, only cells that are present in at least 10
                frames are included in the analysis. If you have any unexpected results
                it could be due to this.
                """
    )
    st.markdown("## Upload raw images")
    raw_images = st.file_uploader(
        "Upload images",
        type=['tiff', 'tif', 'jpg', 'jpeg', 'TIFF', 'TIF', 'JPEG', 'JPG'],
        accept_multiple_files=True
     )

    # Button to start processing
    if len(raw_images) > 0:
        # Validate that the folder contains images
        st.info(f"Uploaded {len(raw_images)} images.")
        st.markdown("## Parameters")
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            save_rois = st.toggle("Keep ROIs?", value=True)
            save_masks = st.toggle("Keep CellPose masks?", value=True)
            save_frame_features = st.toggle("Keep CellPhe frame-features?", value=True)
            save_trackmate_features = st.toggle("Keep TrackMate features?", value=True)
        with col2:
            cellpose_model = st.selectbox(
                "Choose a cellpose segmentation model",
                ("cyto3", "ioLight", "LiveCyte Brightfield"),
            )
            # Ideally would have 20 frames per cell minimum, otherwise time-series
            # features struggle to estimate
            min_frames = st.number_input(
                "Minimum number of frames a cell must be in to be kept",
                min_value=0,
                max_value=len(raw_images),
                value=min(len(raw_images), 20),
            )
            frame_rate = st.number_input(
                "Time period between frames (only used to provide a meaningful unit for cell velocity)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
            )
        st.markdown("## Resuming previous processing")
        st.write("Upload previously calculated intermediate files (masks, ROIs, trackmate feature to skip out processing steps. NB: ensure that the intermediate files are for the same raw images!")
        uploaded_masks = st.file_uploader(
            "Upload previously segmented masks",
            type=['tiff', 'tif', 'jpg', 'jpeg', 'TIFF', 'TIF', 'JPEG', 'JPG'],
            accept_multiple_files=True
         )
        uploaded_roi = st.file_uploader(
            "Upload previously tracked ROI archive",
            type=['zip'],
            accept_multiple_files=False
         )
        uploaded_trackmate_csv = st.file_uploader(
            "Upload previously run trackmate CSV",
            type=['csv'],
            accept_multiple_files=False
         )

        st.markdown("## Run")
        if st.button("Process Images"):
            # Call the process_images function (Assuming it is defined elsewhere in your code)
            ts_variables = process_images(
                raw_images,
                keep_masks=save_masks,
                keep_rois=save_rois,
                keep_trackmate_features=save_trackmate_features,
                keep_cellphe_frame_features=save_frame_features,
                min_frames=min_frames,
                framerate=frame_rate,
                cellpose_model=cellpose_model,
                uploaded_masks=uploaded_masks,
                uploaded_roi=uploaded_roi,
                uploaded_trackmate_csv=uploaded_trackmate_csv
            )

            if ts_variables is None or ts_variables.empty:
                st.write("No time series features extracted.")
            else:
                st.write("Time series feature extraction completed.")
                output_fn = f"cellphe_outputs_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
                with open(ARCHIVE_FN, "rb") as file:
                    st.download_button(
                        label="Download features",
                        data=file,
                        file_name=output_fn,
                        mime="application/zip",
                    )

# Tab 2: Single Population Characterisation
with tab2:
    st.markdown("# Single Population Temporal Characterisation")
    st.markdown(
        "Analysis a single population's temporal characteristics, as obtained by the `cell_features()` function."
    )

    # Allow user to upload a CSV file containing cell features
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to proceed.")
    else:
        # Read the uploaded CSV file
        new_features_df = pd.read_csv(uploaded_file)

        # Ensure that the file contains valid data
        if new_features_df.empty or "CellID" not in new_features_df.columns:
            st.error("Invalid file or missing 'CellID' column.")
        elif "FrameID" not in new_features_df.columns:
            st.error(
                "Upload a feature set of the cells on each frame as output by the cell_features() function in CellPhe."
            )
        else:
            # Dropdown for CellID
            col1, col2 = st.columns(2)
            with col1:
                cell_id = st.selectbox(
                    "Select CellID", new_features_df["CellID"].unique()
                )
            with col2:
                # Exclude 'FrameID' and 'roi_filename' from the dropdown
                selected_feature = st.selectbox(
                    "Select Feature",
                    [
                        col
                        for col in new_features_df.columns
                        if col not in EXCLUDE_ANALYSIS_COLUMNS
                    ],
                )
            # Filter the dataframe based on the selected CellID
            # Plot time-series and densities
            col1, col2 = st.columns(2, vertical_alignment="center")
            with col1:
                cell_data = new_features_df[new_features_df["CellID"] == cell_id]
                # Plot the selected feature against FrameID (line plot)
                if "FrameID" in cell_data.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(
                        cell_data["FrameID"], cell_data[selected_feature], linewidth=3
                    )
                    plt.title(
                        f"Time Series of {selected_feature} for CellID {cell_data['CellID'].values[0]}"
                    )
                    plt.grid(False)  # Remove the grid
                    st.pyplot(plt.gcf())
                else:
                    st.warning("FrameID column not found in data.")

            with col2:
                n_frames = new_features_df["FrameID"].unique().size
                color_map = plt.cm.viridis(np.linspace(1, 0, n_frames))
                plt.figure(figsize=(10, 6))
                new_features_df["FrameID_rev"] = -1 * new_features_df["FrameID"]
                sns.kdeplot(
                    new_features_df,
                    x=selected_feature,
                    hue="FrameID_rev",
                    linewidth=1,
                    palette=color_map,
                    legend=False,
                    common_norm=False,
                )
                plt.title(
                    f"Density Plot of {selected_feature} by Frame"
                )  # Changed to "Frame"
                plt.xlabel(selected_feature)
                plt.ylabel("Density")

                # Add a color bar using the Viridis color map
                norm = plt.Normalize(0, n_frames - 1)
                sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                sm.set_array([])  # Only needed for Matplotlib 3.1 and later

                # Create colorbar using the current axes
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_ticks(
                    np.linspace(0, n_frames - 1, 5)
                )  # Set to fewer intervals, e.g., 5
                cbar.set_ticklabels(
                    [int(i) for i in np.linspace(0, n_frames - 1, 5)]
                )  # Customize tick labels
                cbar.set_label("Frame")  # Changed to "Frame"
                st.pyplot(plt.gcf())

# Tab 3: PCA & Separation Scores
with tab3:
    st.markdown("# PCA and Separation Scores")
    st.markdown(
        """Select the number of groups you want to analyze. They will be
                compared on their time-series features (as output by the
                `time_series_features` function in CellPhe)."""
    )

    # Allow user to select the number of groups
    num_groups = st.number_input(
        "Enter the Number of Groups", min_value=2, value=2, step=1
    )
    st.divider()
    dataframes = [None] * num_groups
    labels = []

    # Input fields for each group
    for i in range(num_groups):
        label_key = f"group_name_{i}"
        if label_key not in st.session_state:
            group_name = ""
        else:
            group_name = st.session_state[label_key]
        st.markdown(f"## Group {i+1}: {group_name}")
        st.text_input(f"Enter name for Group {i+1}", "", key=f"group_name_{i}")

        num_files = st.number_input(
            "Enter the Number of CSV files for this group", min_value=1, value=1, step=1,
            key=f"n_files_group_{i}"
        )
        group_data = [None] * num_files
        for j in range(num_files):
            uploaded_file = st.file_uploader(
                f"Upload CSV File for Group {i+1}", type=["csv"],
                key=f"file_group_{i}_{j}"
            )
            if uploaded_file is None:
                st.info("Please upload a CSV file to proceed.")
            else:
                # Read the uploaded CSV file
                new_features_df = pd.read_csv(uploaded_file)

                # Ensure that the file contains valid data
                if new_features_df.empty or "CellID" not in new_features_df.columns:
                    st.error("Invalid file or missing 'CellID' column.")
                elif "FrameID" in new_features_df.columns:
                    st.error(
                        "Upload a feature set of the cells temporal summary measures, as output by the time_series_features() function in CellPhe."
                    )
                else:
                    new_features_df = new_features_df.drop("CellID", axis=1)
                    group_data.append(new_features_df)
        # Combine all the CSVs for this group
        if sum(x is not None for x in group_data) == num_files:
            group_data = pd.concat(group_data)
            labels.extend(
                [st.session_state[f"group_name_{i}"]] * len(group_data)
            )
            dataframes[i] = group_data
        st.divider()

    # Calculate separation scores if all dataframes are processed
    if sum(x is not None for x in dataframes) == num_groups:
        # Confirm have entered labels for all groups
        input_labels = [st.session_state[f"group_name_{i}"] for i in range(num_groups)]
        if any(x == "" for x in input_labels):
            st.warning("Please enter a name for every group")
        else:
            combined_data = pd.concat(dataframes, ignore_index=True)
            # Calculate separation scores for all groups at once
            sep = cellphe.separation.calculate_separation_scores(dataframes)

            # Sort the separation scores in descending order by the 'Separation' column
            sep_sorted = sep.sort_values(by="Separation", ascending=False)

            # Slider for selecting the top number of discriminatory features to display
            most_discriminatory = cellphe.separation.optimal_separation_features(
                sep_sorted
            )
            num_most_discriminatory = len(most_discriminatory)

            st.markdown("## Exploratory plots")
            st.write(
                "Rather than using all the available 1111 features, which contain can overlapping information, it can sometimes be useful to restrict analysis to a smaller subset of the features. Changing this slider will affect the number of features used in all downstream analyses, including: plotting the separation scores, the PCA plot, classifying new cells."
            )
            n = st.slider(
                "Number of features", 1, len(sep_sorted), num_most_discriminatory
            )

            # Get the top n separation scores for display and analysis
            top_sep_df = sep_sorted.head(n)
            train_features = top_sep_df["Feature"]
            all_features = combined_data.columns

            # Display PCA and separation scores side by side
            plot_sep_and_pca_side_by_side(
                top_sep_df, train_features, combined_data, labels
            )

            # Dropdown to select feature for the boxplot
            st.write(
                f"Compare the {num_groups} groups across features by displaying boxplots for a selected feature."
            )
            col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")
            with col1:
                selected_feature = st.selectbox(
                    "Select Feature for Boxplot", all_features
                )
            with col2:
                plot_boxplot_with_points(combined_data, selected_feature, labels)

            st.divider()
            st.markdown("# Classification of new cells")
            st.write(
                "Upload a test dataset for classification to see how the cells compare to the training data. This must have the same columns as the training data."
            )
            test_file = st.file_uploader("Upload Test CSV File", type=["csv"])

            if test_file is not None:
                test_df = pd.read_csv(test_file)
                test_df = test_df.dropna()
                if test_df.empty or not all(
                    col in test_df.columns for col in train_features
                ):
                    st.error(
                        "The test dataset is invalid or does not contain the necessary features."
                    )
                else:
                    # Use n directly for the number of features for classification
                    # Select top n features from each dataframe for training
                    train_x = combined_data[train_features]
                    train_y = np.array(labels)

                    # Remove rows with NaN values from the training data
                    train_x = train_x.dropna()
                    train_y = train_y[train_x.index]

                    # Select the same features for the test data
                    test_x = test_df[train_features].iloc[:, :n]

                    # Classify the cells using the cleaned training data
                    test_df["Predicted"] = cellphe.classification.classify_cells(
                        train_x, train_y, test_x
                    )

                    # Display pie chart of predicted class distribution if user does not have true labels
                    st.write("Distribution of predicted cell classes:")
                    pie_data = test_df["Predicted"].value_counts()
                    plt.figure(figsize=(8, 6))
                    plt.pie(
                        pie_data,
                        labels=pie_data.index,
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    st.pyplot(plt.gcf())

                    st.write(
                        "Use the dropdown below to investigate the differences in the features between the predicted classes for the test set."
                    )
                    col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")
                    with col1:
                        selected_feature_test = st.selectbox(
                            "Select Feature for test set Boxplot", all_features
                        )
                    with col2:
                        st.write("Boxplot for Selected Feature:")
                        plot_boxplot_with_points(
                            test_df, selected_feature_test, test_df["Predicted"]
                        )
    else:
        st.warning("Please upload a valid CSV for every group.")

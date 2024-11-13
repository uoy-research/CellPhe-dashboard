import time
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import cellphe
import matplotlib.colors as mcolors
from cellphe import segment_images, track_images, cell_features, import_data, time_series_features
import tkinter as tk
from tkinter import filedialog

# Setup tkinter (only used for folder selector)
root = tk.Tk()
# Don't show hidden files/dirs in the file dialog
# Taken from: https://stackoverflow.com/a/54068050/1020006
try:
    # call a dummy dialog with an impossible option to initialize the file
    # dialog without really getting a dialog window; this will throw a
    # TclError, so we need a try...except :
    try:
        root.tk.call('tk_getOpenFile', '-foobarbaz')
    except tk.TclError:
        pass
    # now set the magic variables accordingly
    root.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
    root.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
except:
    pass
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

# Suppress warnings
warnings.filterwarnings('ignore')

EXCLUDE_ANALYSIS_COLUMNS = ['CellID', 'FrameID', 'ROI_filename']

def number_tifs_in_folder(folder: str):
    """
    Checks whether there is at least one TIF in the specified folder.

    :param folder: The folder to check.
    :return: A boolean where True indicates at least one TIF is in the
    directory.
    """
    contents = [x for x in os.listdir(image_folder) if
                os.path.isfile(os.path.join(image_folder, x))]
    extensions = [os.path.splitext(x)[1] for x in contents]
    return sum(x == '.tif' for x in extensions)

# Function to assign feature categories based on substrings in the feature names
def assign_color(feature, colour_mapping):
    # Check for substrings to assign to a group
    if 'Vol' in feature or 'Rad' in feature or 'Wid' in feature or 'Len' in feature or 'Area' in feature:
        return colour_mapping['size']
    elif 'Sph' in feature or 'Box' in feature or 'Rect' in feature or 'VfC' in feature or 'Cur' in feature or 'A2B' in feature or 'Poly' in feature:
        return colour_mapping['shape']
    elif 'FO' in feature or 'Cooc' in feature or 'IQ' in feature:
        return colour_mapping['texture']
    elif feature == 'x' or feature == 'y' or 'trajArea' in feature or 'Vel' in feature or 'Dis' in feature or 'D2T' in feature or 'Trac' in feature:
        return colour_mapping['movement']
    else:
        return colour_mapping['density']

def process_images(image_folder, framerate=0.0028):
    """
    Process a folder of images to extract cell features and time series features.

    Parameters:
    - image_folder: str, path to the folder containing images
    - framerate: float, frame rate for time series analysis (default: 0.0028)

    Returns:
    - tsvariables: DataFrame, extracted time series features (if applicable)
    """
    
    # Step 1: Define paths
    masks_folder = os.path.join(image_folder, "../", os.path.basename(image_folder) + "_masks")
    masks_folder = os.path.abspath(masks_folder)
    
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)

    # TODO should these be temporary folders? or user configurable? Ultimately
    # the user doesn't need to keep the ROIs and Masks

    tracked_csv = os.path.join(image_folder, "../", os.path.basename(image_folder) + "_tracked.csv")
    tracked_csv = os.path.abspath(tracked_csv)
    
    rois_folder = os.path.join(image_folder, "../", os.path.basename(image_folder) + "_rois")
    rois_folder = os.path.abspath(rois_folder)
    
    if not os.path.exists(rois_folder):
        os.makedirs(rois_folder)

    # Step 2: Segment images
    st.write("Segmenting cells...")
    segment_images(image_folder, masks_folder)
    st.success("Segmentation completed.")

    # Step 3: Track images
    st.write("Tracking cells...")
    track_images(masks_folder, tracked_csv, rois_folder)
    st.success("Tracking completed.")

    # Step 4: Import tracked data
    #st.write("Importing tracked data...")
    feature_table = import_data(tracked_csv, "Trackmate_auto")
    #st.success("Tracked data imported.")

    # Step 5: Extract features
    st.write("Extracting CellPhe features...")
    new_features = cell_features(feature_table, rois_folder, image_folder, framerate=framerate)
    st.success("CellPhe feature extraction completed.")

    # Check the counts of tracked frames for each cell
    cell_counts = new_features['CellID'].value_counts()

    # Step 6: Extract time series features only for cells tracked for more than 5 frames
    if any(cell_counts > 5):
        st.write("Extracting time series features for cells tracked for more than 5 frames...")
        
        # Filter new_features to include only cells with more than 3 frames
        valid_cells = cell_counts[cell_counts > 3].index
        filtered_features = new_features[new_features['CellID'].isin(valid_cells)]
        
        # Extract time series features
        tsvariables = time_series_features(filtered_features)
        
        # Save the new features to a CSV file
        features_csv = os.path.join(image_folder, "../", os.path.basename(image_folder) + "_tsvariables.csv")
        tsvariables.to_csv(features_csv, index=False)
        st.success(f"Time series feature extraction completed. CSV saved as: {features_csv}")
    else:
        st.write("No cells tracked for more than 5 frames. Skipping time series feature extraction.")
        tsvariables = pd.DataFrame()  # Return an empty DataFrame if no valid cells exist

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
        'size': 'black',                # For size-related features
        'shape': 'mediumturquoise',     # For shape-related features
        'texture': 'mediumpurple',      # For texture-related features (example)
        'movement': 'cornflowerblue',   # For movement-related features
        'density': 'hotpink'            # For density-related features
    }
    
    # Apply color assignment to the DataFrame based on the feature names
    sep_df['Colour'] = sep_df['Feature'].apply(assign_color, colour_mapping=colour_mapping)

    # Create two columns for side-by-side plotting
    col1, col2 = st.columns(2, vertical_alignment="center")

    # Column 1: Separation Scores
    with col1:
        plt.figure(figsize=(6, 6))  # Set equal size for the separation scores plot
        sns.barplot(data=sep_df, x='Feature', y='Separation', palette=sep_df['Colour'].tolist())  # Convert to list
        plt.title("Separation Scores for Each Feature")
        plt.xlabel("Feature Names")
        plt.ylabel("Separation Score")
        plt.xticks(rotation=90)  # Rotate feature names for better visibility

        # Creating a legend
        handles = []
        for label, color in colour_mapping.items():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                      markerfacecolor=color, markersize=10))
        plt.legend(title='Feature Categories', handles=handles)

        st.pyplot(plt.gcf())

    # Column 2: PCA Plot
    with col2:
        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[top_features])

        # Create a DataFrame with the scaled data for easier handling
        scaled_df = pd.DataFrame(scaled_data, columns=top_features)
        scaled_df['Group'] = labels

        # Drop rows with NaN values before PCA
        scaled_df = scaled_df.dropna()

        # Adjust labels to match the remaining rows after dropping NaNs
        labels_cleaned = scaled_df['Group'].values
        scaled_data_cleaned = scaled_df[top_features].values

        # Perform PCA on the cleaned, scaled data
        pca = PCA(n_components=2)  # Use 2 principal components for visualization
        pca_scores = pca.fit_transform(scaled_data_cleaned)

        # Create a DataFrame for PCA scores
        pca_df = pd.DataFrame(pca_scores, columns=['PC1', 'PC2'])
        pca_df['Group'] = labels_cleaned

        # Plot PCA scores
        plt.figure(figsize=(6, 6))  # Set equal size for the PCA plot
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Group', palette="tab10", s=100)

        plt.title("PCA Scores Plot")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)")
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
    plot_data = pd.DataFrame({'Feature': data[feature], 'Group': labels})

    # Get color palette from 'tab10'
    palette = sns.color_palette("tab10")

    # Create the box plot without transparency first
    boxplot = sns.boxplot(x='Group', y='Feature', data=plot_data, palette=palette, showfliers=False, width=0.5)

    # Set the transparency for the boxes
    for patch in boxplot.artists:
        patch.set_alpha(0.1)  # Set transparency for the boxes

    # Adjust the color palette to be slightly darker for the points
    darker_palette = [mcolors.to_rgba(c, alpha=1) for c in palette]

    # Overlay individual data points, slightly darker and larger than the boxes
    sns.stripplot(x='Group', y='Feature', data=plot_data, palette=darker_palette, 
                  alpha=0.8, size=8)  # Increase point size and darken

    plt.title(f'Box Plot for {feature}')
    plt.xlabel('Group')
    plt.ylabel(feature)

    # Display the plot
    st.pyplot(plt.gcf())

# Streamlit app structure
st.title("The CellPhe Toolkit for Cell Phenotyping")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Image Processing", "Single Population", "Multiple Populations"])

if 'image_folder' not in st.session_state:
    image_folder = ' '
else:
    image_folder = st.session_state.image_folder

# Tab 1: Image Processing
with tab1:
    st.markdown("# Image Processing")
    col1, col2 = st.columns([0.3, 0.7], vertical_alignment='center')
    with col1:
        clicked = st.button("Select image folder")
    with col2:
        st.markdown(f"Selected folder: `{image_folder}`")
    if clicked:
        st.session_state.image_folder = filedialog.askdirectory(master=root)
        st.rerun()


    # Button to start processing
    if image_folder != ' ':
        # Validate that the folder contains images
        if (n_tifs := number_tifs_in_folder(image_folder)) == 0:
            st.warning("No TIFs found in folder")
        else:
            st.info(f"Found {n_tifs} images.")
            if st.button("Process Images"):
                st.write(f"Processing images from folder: {image_folder}")
                # Call the process_images function (Assuming it is defined elsewhere in your code)
                ts_variables = process_images(image_folder)

                if not ts_variables.empty:
                    st.write("Time series feature extraction completed.")
                else:
                    st.write("No time series features extracted.")

# Tab 2: Single Population Characterisation
with tab2:
    st.markdown("# Single Population Temporal Characterisation")
    st.markdown("Analysis a single population's temporal characteristics, as obtained by the `cell_features()` function.")

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
            st.error("Upload a feature set of the cells on each frame as output by the cell_features() function in CellPhe.")
        else:
            # Store the data in session state to retain across interactions
            # TODO needed? never seems to be accessed
            st.session_state['new_features_df'] = new_features_df
            
            # Dropdown for CellID
            col1, col2 = st.columns(2)
            with col1:
                cell_id = st.selectbox("Select CellID", new_features_df["CellID"].unique())
            with col2:
                # Exclude 'FrameID' and 'roi_filename' from the dropdown
                selected_feature = st.selectbox("Select Feature", [col for col in new_features_df.columns if col not in EXCLUDE_ANALYSIS_COLUMNS])
            # Filter the dataframe based on the selected CellID
            # Plot time-series and densities
            col1, col2 = st.columns(2, vertical_alignment="center")
            with col1:
                cell_data = new_features_df[new_features_df["CellID"] == cell_id]
                # Plot the selected feature against FrameID (line plot)
                if "FrameID" in cell_data.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(cell_data["FrameID"], cell_data[selected_feature], linewidth=3)
                    plt.title(f"Time Series of {selected_feature} for CellID {cell_data['CellID'].values[0]}")
                    plt.grid(False)  # Remove the grid
                    st.pyplot(plt.gcf())
                else:
                    st.warning("FrameID column not found in data.")

            with col2:
                n_frames = new_features_df['FrameID'].unique().size
                color_map = plt.cm.viridis(np.linspace(1, 0, n_frames))
                plt.figure(figsize=(10, 6))
                new_features_df['FrameID_rev'] = -1 * new_features_df['FrameID']
                sns.kdeplot(new_features_df,
                            x=selected_feature,
                            hue='FrameID_rev',
                            linewidth=1,
                            palette=color_map,
                            legend=False,
                            common_norm=False
                           )
                plt.title(f'Density Plot of {selected_feature} by Frame')  # Changed to "Frame"
                plt.xlabel(selected_feature)
                plt.ylabel("Density")

                # Add a color bar using the Viridis color map
                norm = plt.Normalize(0, n_frames - 1)
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])  # Only needed for Matplotlib 3.1 and later

                # Create colorbar using the current axes
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_ticks(np.linspace(0, n_frames - 1, 5))  # Set to fewer intervals, e.g., 5
                cbar.set_ticklabels([int(i) for i in np.linspace(0, n_frames - 1, 5)])  # Customize tick labels
                cbar.set_label('Frame')  # Changed to "Frame"
                st.pyplot(plt.gcf())

# Tab 3: PCA & Separation Scores
with tab3:
    st.markdown("# PCA and Separation Scores")
    st.markdown("""Select the number of groups you want to analyze. They will be
                compared on their time-series features (as output by the
                `time_series_features` function in CellPhe).""")

    # Allow user to select the number of groups
    num_groups = st.number_input("Enter the Number of Groups", min_value=2, max_value=10, value=2, step=1)
    st.divider()
    dataframes = []
    labels = []

    # Input fields for each group
    for i in range(num_groups):
        label_key = f"group_name_{i}"
        if label_key not in st.session_state:
            group_name = ""
        else:
            group_name = st.session_state[label_key]
        st.markdown(f"## Group {i+1}: {group_name}")
        st.text_input(f"Enter name for Group {i+1}", "",
                      key=f"group_name_{i}")
        uploaded_file = st.file_uploader(f"Upload CSV File for Group {i+1}", type=["csv"], key=f"group_{i}")
        if uploaded_file is None:
            st.info("Please upload a CSV file to proceed.")
        else:
            # Read the uploaded CSV file
            new_features_df = pd.read_csv(uploaded_file)

            # Ensure that the file contains valid data
            if new_features_df.empty or "CellID" not in new_features_df.columns:
                st.error("Invalid file or missing 'CellID' column.")
            elif "FrameID" in new_features_df.columns:
                st.error("Upload a feature set of the cells temporal summary measures, as output by the time_series_features() function in CellPhe.")
            else:
                new_features_df = new_features_df.drop('CellID', axis=1)
                dataframes.append(new_features_df)
                labels.extend([st.session_state[f"group_name_{i}"]] * len(new_features_df))
        st.divider()

    # Calculate separation scores if all dataframes are processed
    if len(dataframes) == num_groups:
        # Confirm have entered labels for all groups
        input_labels = [st.session_state[f"group_name_{i}"] for i in range(num_groups)]
        if any(x == '' for x in input_labels):
            st.warning("Please enter a name for every group")
        else:
            combined_data = pd.concat(dataframes, ignore_index=True)
            # Calculate separation scores for all groups at once
            sep = cellphe.separation.calculate_separation_scores(dataframes)

            # Sort the separation scores in descending order by the 'Separation' column
            sep_sorted = sep.sort_values(by='Separation', ascending=False)

            # Slider for selecting the top number of discriminatory features to display
            most_discriminatory = cellphe.separation.optimal_separation_features(sep_sorted)
            num_most_discriminatory = len(most_discriminatory)

            
            st.markdown("## Exploratory plots")
            st.write("Rather than using all the available 1111 features, which contain can overlapping information, it can sometimes be useful to restrict analysis to a smaller subset of the features. Changing this slider will affect the number of features used in all downstream analyses, including: plotting the separation scores, the PCA plot, classifying new cells.")
            n = st.slider(
                "Number of features",
                1, len(sep_sorted), num_most_discriminatory
            )

            # Get the top n separation scores for display and analysis
            top_sep_df = sep_sorted.head(n)
            train_features = top_sep_df['Feature']
            all_features = combined_data.columns

            # Display PCA and separation scores side by side
            plot_sep_and_pca_side_by_side(
                top_sep_df,
                train_features,
                combined_data,
                labels
            )

            # Dropdown to select feature for the boxplot
            st.write(f"Compare the {num_groups} groups across features by displaying boxplots for a selected feature.")
            col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")
            with col1:
                selected_feature = st.selectbox("Select Feature for Boxplot", all_features)
            with col2:
                plot_boxplot_with_points(combined_data, selected_feature, labels)

            st.divider()
            st.markdown("# Classification of new cells")
            st.write("Upload a test dataset for classification to see how the cells compare to the training data. This must have the same columns as the training data.")
            test_file = st.file_uploader("Upload Test CSV File", type=["csv"])

            if test_file is not None:
                test_df = pd.read_csv(test_file)
                test_df = test_df.dropna()
                if test_df.empty or not all(col in test_df.columns for col in train_features):
                    st.error("The test dataset is invalid or does not contain the necessary features.")
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
                    test_df['Predicted'] = cellphe.classification.classify_cells(train_x, train_y, test_x)[:, 3]

                    # Display pie chart of predicted class distribution if user does not have true labels
                    st.write("Distribution of predicted cell classes:")
                    pie_data = test_df['Predicted'].value_counts()
                    plt.figure(figsize=(8, 6))
                    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                    st.pyplot(plt.gcf())

                    st.write("Use the dropdown below to investigate the differences in the features between the predicted classes for the test set.")
                    col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")
                    with col1:
                        selected_feature_test = st.selectbox("Select Feature for test set Boxplot", all_features)
                    with col2:
                        st.write("Boxplot for Selected Feature:")
                        plot_boxplot_with_points(test_df, selected_feature_test, test_df['Predicted'])
    else:
        st.warning("Please upload a valid CSV for every group.")

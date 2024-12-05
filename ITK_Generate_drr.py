#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:01:15 2024

@author: harrysmith
"""
import itk
import numpy as np
import os

dicom_dir = "/Users/harrysmith/Documents/Carry_lab/COPDGene_Imaging/10002K_INSP_B31f_267_COPD1/dicom"

def read_dicom_series(dicom_dir):
    """
    Reads a DICOM series from the specified directory using ITK.
    """
    # Generate file names for the DICOM series
    dicom_names_generator = itk.GDCMSeriesFileNames.New()
    dicom_names_generator.SetDirectory(dicom_dir)

    # Get list of series IDs
    series_ids = dicom_names_generator.GetSeriesUIDs()
    if not series_ids:
        raise ValueError(f"No DICOM series found in the directory: {dicom_dir}")
    
    # Debug: Print available series IDs
    print(f"Available Series IDs: {series_ids}")
    
    # Use the first series ID
    selected_series_id = series_ids[0]

    # Get file names for the selected series
    dicom_file_names = dicom_names_generator.GetFileNames(selected_series_id)
    if not dicom_file_names:
        raise ValueError(f"No files found for series ID: {selected_series_id}")

    # Debug: Print file names being passed
    print(f"File names for series {selected_series_id}: {dicom_file_names}")
    
    # Define the image type (3D, float pixel type)
    PixelType = itk.F  # Floating point
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]

    # Initialize the reader with the image type
    reader = itk.ImageSeriesReader[ImageType].New()
    reader.SetFileNames(dicom_file_names)
    
    # Read the DICOM series
    reader.Update()  # Trigger the pipeline to process the data
    image = reader.GetOutput()  # Retrieve the processed image

    # Debug: Print image size
    print(f"DICOM series loaded with size: {image.GetLargestPossibleRegion().GetSize()}")
    
    return image


'''
def generate_drr(input_image, projection_matrix=None):
    """
    Generates a 2D DRR image from a 3D DICOM image using ITK.
    Optionally, a projection matrix can be applied for custom orientations.
    """
    # For simplicity, use maximum intensity projection (MIP) as a placeholder
    # Projection matrix implementation will depend on your specific use case
    np_image = itk.GetArrayFromImage(input_image)

    # Generate DRR via max intensity projection
    drr_image_array = np.max(np_image, axis=1)  # Project along Z-axis

    # Convert DRR back to ITK Image
    drr_image = itk.image_from_array(drr_image_array.astype(np.float32))
    return drr_image
'''
"""
def generate_drr(input_image, projection_axis=0, smooth_sigma=1, enhance_contrast=True):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import cv2

    np_image = itk.GetArrayFromImage(input_image)

    # Normalize the intensity
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))

    # Apply Gaussian smoothing
    if smooth_sigma > 0:
        np_image = gaussian_filter(np_image, sigma=smooth_sigma)

    # Generate DRR via max intensity projection
    drr_image_array = np.max(np_image, axis=projection_axis)

    # Enhance contrast
    if enhance_contrast:
        drr_image_array = (drr_image_array - np.min(drr_image_array)) / (np.max(drr_image_array) - np.min(drr_image_array))
        drr_image_array = cv2.equalizeHist((drr_image_array * 255).astype(np.uint8))

    # Convert DRR back to ITK Image
    drr_image = itk.image_from_array(drr_image_array.astype(np.float32))
    return drr_image
"""

import itk
import numpy as np
import cv2

def generate_drr(input_image, projection_matrix=None, focal_shift=0, projection_axis=1, slice_ranges={"y": (300, 330)}):
    """
    Generates a 2D DRR image from a 3D DICOM image using ITK, allowing customization of focal point and slicing along any axis.
    
    Parameters:
        input_image: itk.Image
            The input 3D DICOM image.
        projection_matrix: np.array, optional
            Projection matrix for custom orientations (not implemented in this example).
        focal_shift: int
            Shift along the projection axis to move the focal point anteriorly/posteriorly.
        projection_axis: int
            Axis along which to perform the projection (0 = X, 1 = Y, 2 = Z).
        slice_ranges: dict, optional
            Dictionary specifying slice ranges for each axis (keys: "x", "y", "z").
            Example: {"x": (start_x, end_x), "y": (start_y, end_y), "z": (start_z, end_z)}
    """
    np_image = itk.GetArrayFromImage(input_image)

    # Apply focal shift by cropping or padding the 3D array
    if focal_shift != 0:
        if focal_shift > 0:
            if projection_axis == 0:  # X-axis
                np_image = np.pad(np_image, ((focal_shift, 0), (0, 0), (0, 0)), mode='constant')
            elif projection_axis == 1:  # Y-axis
                np_image = np.pad(np_image, ((0, 0), (focal_shift, 0), (0, 0)), mode='constant')
            elif projection_axis == 2:  # Z-axis
                np_image = np.pad(np_image, ((0, 0), (0, 0), (focal_shift, 0)), mode='constant')
        else:
            if projection_axis == 0:  # X-axis
                np_image = np_image[abs(focal_shift):, :, :]
            elif projection_axis == 1:  # Y-axis
                np_image = np_image[:, abs(focal_shift):, :]
            elif projection_axis == 2:  # Z-axis
                np_image = np_image[:, :, abs(focal_shift):]

    # Apply axis-specific slicing if provided
    if slice_ranges:
        if "x" in slice_ranges:
            start_x, end_x = slice_ranges["x"]
            np_image = np_image[start_x:end_x, :, :]
        if "y" in slice_ranges:
            start_y, end_y = slice_ranges["y"]
            np_image = np_image[:, start_y:end_y, :]
        if "z" in slice_ranges:
            start_z, end_z = slice_ranges["z"]
            np_image = np_image[:, :, start_z:end_z]

    # Generate DRR via max intensity projection along the specified axis
    drr_image_array = np.max(np_image, axis=projection_axis)

    # Enhance contrast using CLAHE
    drr_image_array = cv2.normalize(drr_image_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    drr_image_array = drr_image_array.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    drr_image_array = clahe.apply(drr_image_array)

    # Apply Sobel edge detection for gradient enhancement
    sobel_x = cv2.Sobel(drr_image_array, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal gradients
    sobel_y = cv2.Sobel(drr_image_array, cv2.CV_64F, 0, 1, ksize=5)  # Vertical gradients
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Combine gradients

    # Normalize gradient image for better visualization
    sobel_combined = cv2.normalize(sobel_combined, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    sobel_combined = sobel_combined.astype(np.uint8)

    # Combine original DRR with gradient-enhanced version
    combined_image = cv2.addWeighted(drr_image_array, 0.6, sobel_combined, 0.4, 0)

    # Convert final image back to ITK format
    drr_image = itk.image_from_array(combined_image.astype(np.float32))
    return drr_image





def save_image(image, output_path):
    """
    Saves the image to the specified path.
    """
    itk.imwrite(image, output_path)
    print(f"Image saved to {output_path}")

def main():
    # Define paths
    dicom_dir = "/Users/harrysmith/Documents/Carry_lab/COPDGene_Imaging/10002K_INSP_B31f_267_COPD1/dicom"
    output_path = "/Users/harrysmith/Documents/Carry_lab/COPDGene_Imaging/10002K_INSP_B31f_267_COPD1/drr_output.nrrd"

    try:
        # Step 1: Read the DICOM series
        dicom_image = read_dicom_series(dicom_dir)

        # Step 2: Generate DRR
        drr_image = generate_drr(dicom_image)

        # Step 3: Save DRR
        save_image(drr_image, output_path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


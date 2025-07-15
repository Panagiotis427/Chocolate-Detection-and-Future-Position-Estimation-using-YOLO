"""Process image containing Data Matrix code to calculate the real diameter of detected objects on other images."""

import os
import cv2
import numpy as np
from pylibdmtx import pylibdmtx
import matplotlib.pyplot as plt


data_matrix_image_path = "/data_matrix.png"
image_dir_path = "/object_size_measurement/object detection images"
table_input_path = "/object_size_measurement/objects_predicted_true_measurement_table.txt"
table_output_path = "/object_size_measurement/objects_predicted_measurement_table.txt"


def mm_to_px_ratio_by_data_matrix(data_matrix_image_path):
    image = cv2.imread(data_matrix_image_path)
    if image is None:
        print(f"Error: Unable to open image file {data_matrix_image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decoded_objects = pylibdmtx.decode(gray)

    if not decoded_objects:
        print("No Data Matrix code detected.")
        return

    _, _, w, h = decoded_objects[0].rect
    w, h = abs(w), abs(h)

    # use predefined size for the data matrix code
    distance_mm = 10.0 # mm
    
    return distance_mm * 2 / (w + h)


def read_yolo_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        # confidence = round(float(parts[5]),2)
        annotations.append((class_id, x_center, y_center, width, height))

    return annotations


def read_image_draw_annotations(image_path, annotations, mm_to_px_ratio):
    objects_pred_measurement = []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return

    h, w, _ = image.shape

    for annotation in annotations:
        class_id, x_center, y_center, width, height = annotation
        class_id_map = {0: "object"}
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        hypotenuse = np.sqrt(width**2 + height**2)
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
    
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 50), 6)

        # Calculate the diameter of the object
        diameter_px_g_mean = np.sqrt(hypotenuse * max(width,height))
        diameter_mm = round(mm_to_px_ratio * diameter_px_g_mean, 2)
        
        objects_pred_measurement.append((image_path[image_path.rfind('/')+1:-4], class_id, x1, y1, x2, y2, diameter_mm))

        text = str(class_id_map.get(class_id, "Unknown")) + f" {diameter_mm} mm"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
        text_x = x1
        text_y = y1 - 10

        if text_x < 0:
            text_x = 0
        if text_y < text_size[1]:
            text_y = y2 + 10 + text_size[1]
        if text_x + text_size[0] > w:
            text_x = w - text_size[0]
        if text_y > h:
            text_y = h

        # Put blue (255, 100, 50) or red (0, 0, 255) background for the text and white (255, 255, 255) text
        cv2.rectangle(image, (text_x, text_y - text_size[1]+8), (text_x + text_size[0], text_y+8), (255, 100, 50), -1)
        cv2.putText(image, text, (text_x, text_y+8), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)

    cv2.imwrite(image_path.replace("images", "predictions measurements").replace(".png","_measured.png"), image)

    return objects_pred_measurement


def explort_all_objects_measurement_table(true_objects_measurement, pred_objects_measurement, output_path):
    """
    Plots a table of object measurements and saves it as an image.

    Parameters:
    pred_objects_measurement (list): List of tuples containing object measurements.
    output_path (str): Path to save the table image.
    """
    # Define table headers
    headers = ["Image ID", "Class ID, X1, Y1, X2, Y2", "True Diameter (mm)", "Pred Diameter (mm)", "Absolute Error (mm)", "Squared Error (%)"]

    # Define lists to store the errors
    absolute_error_list = []
    squared_error_list = []

    with open(output_path, 'w') as file:
        # Write headers
        file.write(f'{headers[0]}\t\t{headers[1]}\t\t{headers[2]}\t{headers[3]}\t{headers[4]}\t{headers[5]}\n')

        # Format the data for the table
        formatted_data = []
        for index, object in enumerate(pred_objects_measurement):
            image_id = object[0]
            class_id = object[1]
            x1 = object[2]
            y1 = object[3]
            x2 = object[4]
            y2 = object[5]
            pred_diameter_mm = round(object[6], 2)

            # Find the true object measurement
            true_diameter_mm = true_objects_measurement[index][6]

            # Calculate the error
            absolute_error = round(abs(true_diameter_mm - pred_diameter_mm), 2)
            squared_error = round((true_diameter_mm-pred_diameter_mm)**2, 2)
            absolute_error_list.append(absolute_error)
            squared_error_list.append(squared_error)

            if len(f"{x1}, {y1}, {x2}, {y2}") <= 20:
                file.write(f"{image_id}\t{class_id}, {x1}, {y1}, {x2}, {y2}\t\t{true_diameter_mm}\t\t\t{pred_diameter_mm}\t\t\t{absolute_error:.2f}\t\t{squared_error:.2f}\n")
            else:
                file.write(f"{image_id}\t{class_id}, {x1}, {y1}, {x2}, {y2}\t{true_diameter_mm}\t\t\t{pred_diameter_mm}\t\t\t{absolute_error:.2f}\t\t{squared_error:.2f}\n")

            formatted_data.append([image_id, f"{class_id}, {x1}, {y1}, {x2}, {y2}", true_diameter_mm, pred_diameter_mm, absolute_error, squared_error])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, len(formatted_data) * 0.5))  # Adjust the figure size

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=formatted_data, colLabels=headers, cellLoc='center', loc='center')

    # Adjust layout
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Set alternating row colors
    for i in range(len(formatted_data)):
        if i % 2 == 0:
            color = 'lightgrey'
        else:
            color = 'white'
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)

    # Ensure the values in the cells are fully shown
    table.auto_set_column_width(col=list(range(len(headers))))

    # Save the table as an image
    plt.savefig(output_path.replace('.txt', '.png'), bbox_inches='tight')
    plt.close()

    # Calculate the mean absolute error and root mean squared error
    mean_absolute_error = round(sum(absolute_error_list) / len(absolute_error_list), 2)
    root_mean_squared_error = round(np.sqrt(sum(squared_error_list) / len(squared_error_list)), 2)
    print(f"Mean Absolute Error: {mean_absolute_error} mm. Root Mean Squared Error: {root_mean_squared_error} mm.")

    # Write the mean absolute error and mean squared error to the file
    with open(output_path, 'a') as file:
        file.write(f"\nMean Absolute Error: {mean_absolute_error} mm. Root Mean Squared Error: {root_mean_squared_error} mm.")


def read_true_objects_measurement_table(table_path):

    all_true_objects_measurement = []

    with open(table_path, 'r') as file:
        lines = file.readlines()

    for line in lines[1:]:
        parts = line.strip().split()
        image_id = parts[1]
        class_id = int(parts[2].replace(',', ''))
        x1 = int(parts[3].replace(',', ''))
        y1 = int(parts[4].replace(',', ''))
        x2 = int(parts[5].replace(',', ''))
        y2 = int(parts[6])
        diameter_mm = float(parts[7])

        all_true_objects_measurement.append((image_id, class_id, x1, y1, x2, y2, diameter_mm))

    return all_true_objects_measurement


if __name__ == "__main__":

    mm_to_px_ratio = mm_to_px_ratio_by_data_matrix(data_matrix_image_path)

    all_pred_objects_measurement = []

    for image in os.listdir(image_dir_path):

        if image.endswith(".png"):

            image_path = image_dir_path + '/' + image

            pred_path = image_path.replace("images", "predictions").replace(".png", ".txt")

            pred_annotations = read_yolo_annotations(pred_path)
            pred_objects_measurement = read_image_draw_annotations(image_path, pred_annotations, mm_to_px_ratio)

            all_pred_objects_measurement.extend(pred_objects_measurement)


    all_true_objects_measurement = read_true_objects_measurement_table(table_input_path)

    # Plot the table for all images' objects measurement
    explort_all_objects_measurement_table(all_true_objects_measurement, all_pred_objects_measurement, table_output_path)

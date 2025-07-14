"""Program to track and plot objects' (past and future) path."""
#import cProfile
import os
import sys
from datetime import datetime
import random
from collections import defaultdict
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Set current working directory and load control parameters
cwd = os.getcwd()
with open(os.path.join(cwd, 'control_panel.yaml'),'r', encoding="utf-8") as file:
    panel = yaml.safe_load(file)

# Calculate the rest control parameters
video_path = os.path.join(os.path.dirname(cwd), panel["VIDEO_PATH"])
video = cv2.VideoCapture(video_path)
video_handling = {'quit' : [ord("q"), ord("Q"), ord(";"), 27],
                  'pause' : [ord("p"), ord("P"), ord(" ")]}

# Extract selected video's information
if 'conveyor' in video_path:
    SELECTED_VIDEO = f"{round(video.get(3))}x{round(video.get(4))}"
else:
    print('Incorrect video path')
    sys.exit(1)

# [pixel] coordinates of shown conveyor
SELECTED_REGION_CORNERS = panel["SELECTED_REGION_CORNERS"][SELECTED_VIDEO]
SELECTED_REGION_DIMENSIONS = panel["SELECTED_REGION_DIMENSIONS"][SELECTED_VIDEO]

# [fps] of video or camera stream
CAMERA_FPS = round(video.get(5))

# Initialize track of time, positions, etc.
time_instances = []
trails = defaultdict(lambda: [])
preprocess_time = []
inference_time = []
postprocess_time = []
frames = []

# Load the YOLOv8 model
model = YOLO(os.path.join(os.path.dirname(cwd), panel['MODEL_PATH']))


def color_generator(object_id):
    """Function used to generate a specific color for each object's box"""

    random.seed(object_id)
    present_box_color = [0, 0, 0]
    while (abs(present_box_color[0]-present_box_color[1]) < 240 and
           abs(present_box_color[1]-present_box_color[2]) < 240 and
           abs(present_box_color[2]-present_box_color[0]) < 240):
        present_box_color = [random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)]
    return present_box_color


def calculate_time_intervals(message):
    """Function to calculate time intervals."""

    dt = datetime.now()
    time_instances.append((dt.hour*3600+dt.minute*60+dt.second)*10**3+dt.microsecond*10**(-3))
    if len(time_instances) == 2:
        time_interval = time_instances[-1] - time_instances[0]
        if panel["PRINT_TIME_INTERVALS"]:
            print(f"{message} {time_interval:.3f} ms.")
        time_instances.pop(0)
        return time_interval
    return None


def apply_max_ot_capacity(objects_parameter_history):
    """Function to restrict maximum objects being tracked."""

    if len(objects_parameter_history) > panel["MAX_OT_CAPACITY"]:
        for specific_id in list(objects_parameter_history.keys()):
            if (round(objects_parameter_history[specific_id][-1][0], 1) ==
                    round(objects_parameter_history[specific_id][0][0], 1) and
                    round(objects_parameter_history[specific_id][-1][1], 1) ==
                    round(objects_parameter_history[specific_id][0][1], 1)):
                del objects_parameter_history[specific_id]
                break
        else:
            del objects_parameter_history[list(objects_parameter_history.keys())[0]]


def plot_predictions(image, boxes_data):
    """Function used to plot predictions on a frame."""

    # Create copy of image to draw on it
    out = image.copy()

    # Define thickness of line and thickness, font scale of text
    th = min(round(sum(image.shape) * 0.001), 2)
    fs = max(th-1, 0.4)

    # Draw selected region box
    cv2.rectangle(out, SELECTED_REGION_CORNERS[0], SELECTED_REGION_CORNERS[1],
                  panel["SELECTED_REGION_COLOR"], thickness=th*4, lineType=cv2.LINE_AA)

    for box_data in boxes_data:

        # If conf threshold has been set, filter every box under conf threshold and/or without id
        if (panel["CONFIDENCE_THRESHOLD"] and box_data[-2] > panel["CONFIDENCE_THRESHOLD"] and
            len(box_data) == 7):

            # Select system of reference
            if panel["DETECT_ON_SELECTED_REGION"]:
                reference_point = SELECTED_REGION_CORNERS[0]
            else:
                reference_point = (0, 0)
            reduced_box_data = [int(box_data[0] + reference_point[0]),
                                int(box_data[1] + reference_point[1]),
                                int(box_data[2] + reference_point[0]),
                                int(box_data[3] + reference_point[1]),
                                int(box_data[4])]

            # Draw current object's box coordinates
            object_p1_now = [reduced_box_data[0], reduced_box_data[1]]
            object_p2_now = [reduced_box_data[2], reduced_box_data[3]]
            box_color = color_generator(reduced_box_data[-1])
            cv2.rectangle(out, object_p1_now, object_p2_now, box_color,
                          thickness=th, lineType=cv2.LINE_AA)

            # Define label with/without score
            label = f"id:{int(box_data[4])}"
            if panel["SHOW_CLASSES"]:
                label += f", {panel['CLASSES'][int(box_data[-1])]}"
            if panel["SHOW_SCORES"]:
                label += f", conf={round(100 * float(box_data[-2]), 1)}%"
            # Draw current object's label
            # text width, height
            width, height = cv2.getTextSize(label, fontFace=0, fontScale=fs, thickness=th)[0]
            label_p1 = [object_p1_now[0], object_p1_now[1]-height-3]
            label_p2 = [object_p1_now[0]+width, object_p1_now[1]]
            cv2.rectangle(out, label_p1, label_p2, box_color, cv2.FILLED)
            cv2.putText(out, label, (label_p1[0], label_p1[1]+height+1), fontFace=0, fontScale=fs,
                        color=panel["TEXTS_COLOR"], thickness=th, lineType=cv2.LINE_AA)

            # Create trail of the current objects' position
            if panel["SHOW_TRAILS"]:
                trail = trails[reduced_box_data[4]]
                trail.append((round((reduced_box_data[0] + reduced_box_data[2]) / 2),
                              round((reduced_box_data[1] + reduced_box_data[3]) / 2)))
                # Draw the trails
                points = np.hstack(trail).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [points], isClosed=False,
                              color=panel["LINES_COLOR"], thickness=th)

                # Remove residual data
                if len(trail) > CAMERA_FPS*panel["PAST_TIME"]:
                    trail.pop(0)
                if panel["LIMIT_OT_CAPACITY"]:
                    apply_max_ot_capacity(trails)

    return out


# Loop through the video frames
while video.isOpened():
    if panel["PRINT_TIME_INTERVALS"]:
        track_time = calculate_time_intervals("\nAfter last cap.isOpened(), "
                                              "time interval is: ")

    # Read a frame from the video
    ret, color_frame = video.read()
    if panel["PRINT_TIME_INTERVALS"]:
        track_time = calculate_time_intervals("\nAfter last read of image, "
                                              "time interval is: ")

    if ret:
        if panel["DETECT_ON_SELECTED_REGION"]:
            # Select SELECTED_REGION's frame
            detection_frame = color_frame[SELECTED_REGION_CORNERS[0][1]:
                                          SELECTED_REGION_CORNERS[1][1],
                                          SELECTED_REGION_CORNERS[0][0]:
                                          SELECTED_REGION_CORNERS[1][0],
                                          :]
        else:
            # Select whole frame
            detection_frame = color_frame

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(detection_frame, conf=panel["CONFIDENCE_THRESHOLD"],
                              iou=panel["IOU_THRESHOLD"], persist=True, imgsz=640,
                              verbose=False, device=0)
        if panel["PRINT_TIME_INTERVALS"]:
            track_time = calculate_time_intervals("\nAfter last tracking, "
                                                  "time interval is: ")

        if panel["SHOW_TRACKING_PERFORMANCE"] or panel["SAVE_TRACKING_PERFORMANCE"]:
            # Store tracking performance
            preprocess_time.append(results[0].speed['preprocess'])
            inference_time.append(results[0].speed['inference'])
            postprocess_time.append(results[0].speed['postprocess'])

        # Plot the predictions on the frame
        output = plot_predictions(color_frame, results[0].boxes.data)
        if panel["PRINT_TIME_INTERVALS"]:
            plot_predictions_time = calculate_time_intervals("\nAfter last plot predictions, "
                                                             "time interval is: ")

        output = cv2.resize(output, tuple(panel["OUTPUT_RESOLUTION"]))

        if panel["SHOW_VIDEO"]:
            # Resize and display the annotated frame
            cv2.imshow('Object tracking', output)
            if panel["PRINT_TIME_INTERVALS"]:
                display_time = calculate_time_intervals("\nAfter last display of image, "
                                                        "time interval is: ")

        if panel["SAVE_VIDEO"]:
            frames.append(output)
            FRAME_COUNT = len(frames)
            print(f"\rFrame count : {FRAME_COUNT}    Seconds in video : " \
                  f"{round(FRAME_COUNT/CAMERA_FPS,2)}", end="")
            if panel["PRINT_TIME_INTERVALS"]:
                save_time = calculate_time_intervals("\nAfter last save of image, "
                                                        "time interval is: ")

        key = cv2.waitKey(1)
        # Pause the loop if 'p' or 'P' or Space character is pressed
        if key in video_handling["pause"]:
            # wait until any key is pressed
            key = cv2.waitKey(-1)
        # Break the loop if 'q' or 'Q' or Esc character is pressed
        if key in video_handling["quit"]:
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()

if panel["SAVE_VIDEO"]:
    # Save the concatenated video as an MP4 file
    timestamp = datetime.now()
    output_file = f"{str(timestamp)[:-7].replace(' ', '_').replace('-','').replace(':','')}_" +\
                  os.path.splitext(os.path.basename(__file__))[0].replace("_off",".mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path.replace('evaluation_data','early_demos').
                                   replace(video_path[video_path.find("/202"):], "") +\
                                   f"/{output_file}", fourcc, CAMERA_FPS, 
                                   (frames[0].shape[1], frames[1].shape[0]))
    # Write each frame to the video file
    for frame in frames:
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
    print(f"\nVideo saved as : {output_file}")


if panel["SHOW_TRACKING_PERFORMANCE"] or panel["SAVE_TRACKING_PERFORMANCE"]:

    model_name = panel["MODEL_PATH"][(panel["MODEL_PATH"].find("models/")+7):
                                     panel["MODEL_PATH"].find(".pt")].replace("/","_")
    #Set a style to use
    plt.style.use('ggplot')

    # Create a figure
    fig = plt.figure(f'YOLO tracking performance of {model_name}')
    plt1 = fig.add_subplot(311)
    plt2 = fig.add_subplot(312)
    plt3 = fig.add_subplot(313)

    # Plot points on each subplot
    plt1.plot(range(10, len(preprocess_time)+1), preprocess_time[9:], color='blue')
    plt1.set_xlabel("frames")
    plt1.set_ylabel("time (ms)")
    avg_preprocess_time = round(sum(preprocess_time)/len(preprocess_time), 2)
    plt1.axis(ymin=0.2*avg_preprocess_time, ymax=2*avg_preprocess_time)
    plt1.set_title(f"preprocess time, avg = {avg_preprocess_time} ms")

    plt2.plot(range(10, len(inference_time)+1), inference_time[9:], color='green')
    plt2.set_xlabel("frames")
    plt2.set_ylabel("time (ms)")
    avg_inference_time = round(sum(inference_time)/len(inference_time), 2)
    plt2.axis(ymin=0.2*avg_inference_time, ymax=2*avg_inference_time)
    plt2.set_title(f"inference time, avg = {avg_inference_time} ms")

    plt3.plot(range(10, len(postprocess_time)+1), postprocess_time[9:], color='red')
    plt3.set_xlabel("frames")
    plt3.set_ylabel("time (ms)")
    avg_postprocess_time = round(sum(postprocess_time)/len(postprocess_time), 2)
    plt3.axis(ymin=0.2*avg_postprocess_time, ymax=2*avg_postprocess_time)
    plt3.set_title(f"postprocess time, avg = {avg_postprocess_time} ms")

    # Adjust space between subplots
    fig.subplots_adjust(hspace=1.5)

    if panel["SHOW_TRACKING_PERFORMANCE"]:
        # Show the plot
        plt.show()
    else:
        # Save the plot
        plt.savefig(os.path.join(cwd, 'YOLO_tracking_performances', f'{model_name}.jpg'))

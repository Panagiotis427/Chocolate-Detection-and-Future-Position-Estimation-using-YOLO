"""Program to track and plot objects' (past and future) path."""
#import cProfile
import os
import sys
import time
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
video_handling = {'pause' : [ord("p"), ord("P"), ord("π"), ord("Π"), ord(" ")],
                  'forward': [ord("f"), ord("F"), ord("φ"), ord("Φ")],
                  'backward': [ord("b"), ord("B"), ord("β"), ord("Β")],
                  'quit' : [ord("q"), ord("Q"), ord(";"), 27]
                  }

WAIT_FOR_BREAK_LINE = 0
# Extract selected video's information
if 'conveyor' in video_path:
    SELECTED_VIDEO = f"{round(video.get(3))}x{round(video.get(4))}"
else:
    print('Incorrect video path')
    sys.exit(1)

# [pixel] coordinates of shown conveyor
SELECTED_REGION_CORNERS = panel["SELECTED_REGION_CORNERS"][SELECTED_VIDEO]
SELECTED_REGION_DIMENSIONS = panel["SELECTED_REGION_DIMENSIONS"][SELECTED_VIDEO]

# [pixels/meter] scaling factor
SCALING = 1/2*((SELECTED_REGION_CORNERS[1][0]-SELECTED_REGION_CORNERS[0][0])/
               SELECTED_REGION_DIMENSIONS[0]+\
               (SELECTED_REGION_CORNERS[1][1]-SELECTED_REGION_CORNERS[0][1])/
               SELECTED_REGION_DIMENSIONS[1])

# [fps] of video or camera stream
CAMERA_FPS = round(video.get(5))
# minimum number of pixels that is considered as movement
NUMBER_OF_FRAMES = round(panel["MINIMUM_MOVEMENT"]*CAMERA_FPS/(SCALING*panel["MINIMUM_SPEED"]))+1
# [s] elapsed between NUMBER_OF_FRAMES frames
TIME_FRAME = (NUMBER_OF_FRAMES-1)/CAMERA_FPS

# Initialize track of positions, etc.
positions = defaultdict(lambda: [])
past_moves = defaultdict(lambda: [])
speeds = defaultdict(lambda: [])
future_moves = defaultdict(lambda: [])
ids_lists = []
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

def calculate_time_intervals(switch=0, initial_time=0, operation_name=""):
    """Function to calculate time intervals."""

    if switch:
        time_interval = round((time.time()-initial_time)*1000, 3)
        print(f"\nAfter last {operation_name}, time interval is: {time_interval}")
        initial_time = time.time()
        return initial_time, time_interval
    return None, None

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


def calculate_predictions(reduced_box_data):
    """Function used to calculate predictions of the objects' path."""

    # Keep track of position, past_move, speed and future_move
    position = positions[reduced_box_data[4]]
    past_move = past_moves[reduced_box_data[4]]
    speed = speeds[reduced_box_data[4]]
    future_move = future_moves[reduced_box_data[4]]

    # Define default position, past_move, speed and future_move
    for physical_parameter in [position, past_move, speed, future_move]:
        physical_parameter.append((0, 0))
        if len(physical_parameter) >= NUMBER_OF_FRAMES+1:
            physical_parameter.pop(0)
    position[-1] = (round((reduced_box_data[0] + reduced_box_data[2]) / 2),
                    round((reduced_box_data[1] + reduced_box_data[3]) / 2))

    if len(position) == NUMBER_OF_FRAMES:
        new_movement = np.linalg.norm(np.array(position[-1]) -
                                      np.array(position[-NUMBER_OF_FRAMES]))
        if new_movement > panel["MINIMUM_MOVEMENT"]:
            # Distance (x_past_move, y_past_move)
            past_move[-1] = (round((position[-1][0]-position[-NUMBER_OF_FRAMES][0])
                                  / SCALING, 3),
                            round((position[-1][1]-position[-NUMBER_OF_FRAMES][1])
                                  / SCALING, 3))
            if panel["PRINT_CALCULATIONS"]:
                print(f"\nid {reduced_box_data[4]} has moved: {past_move[-1][0]:.3f} m \
                      in x axis and: {past_move[-1][1]:.3f} m in y axis.")

            # Speed magnitude (x_speed, y_speed)
            speed[-1] = [round(past_move[-1][0] / TIME_FRAME, 3),
                                   round(past_move[-1][1] / TIME_FRAME, 3)]
            if panel["PRINT_CALCULATIONS"]:
                print(f"\nid {reduced_box_data[4]} moved at: {speed[-1][0]:.3f} m/s \
                      in x axis and: {speed[-1][1]:.3f} m/s in y axis.")

            if len(speed) >= panel["NUMBER_OF_TERMS"]:
                weight_sum = 0
                term_sum = [0, 0]
                for term in range(1, panel["NUMBER_OF_TERMS"]+1):
                    weight = round((1-term/(panel["NUMBER_OF_TERMS"]+1))**2, 3)
                    weight_sum += weight
                    term_sum[0] += weight*speed[-term][0]
                    term_sum[1] += weight*speed[-term][1]
                speed[-1][0] = round(term_sum[0]/weight_sum, 3)
                speed[-1][1] = round(term_sum[1]/weight_sum, 3)

            # Displacement (x_future_move, y_future_move)
            future_move[-1] = (round(speed[-1][0] * panel["FUTURE_TIME"], 3),
                                round(speed[-1][1] * panel["FUTURE_TIME"], 3))
            if panel["PRINT_CALCULATIONS"]:
                print(f"\nid {reduced_box_data[4]} will move: {future_move[-1][0]:.3f} m \
                      in x axis and: {future_move[-1][1]:.3f} m in y axis.")

            future_move[-1] = (round(future_move[-1][0]*SCALING),
                                round(future_move[-1][1]*SCALING))

        # Remove residual data
        if panel["LIMIT_OT_CAPACITY"]:
            for objects_parameter_history in [positions, past_moves,
                                              speeds, future_moves]:
                apply_max_ot_capacity(objects_parameter_history)

    return position[-1], past_move[-1], speed[-1], future_move[-1]


def plot_predictions(image, boxes_data):
    """Function used to plot predictions on a frame."""

    # Create copy of image to draw on it
    out = image.copy()

    # Define thickness of line and thickness, font scale of text
    th = min(round(sum(image.shape) * 0.001), 2)
    fs = max(th-1, 0.4)

    # Initialize blank mask image of same dimensions for drawing the shapes
    shapes = np.zeros_like(image, np.uint8)

    # Draw selected region box
    cv2.rectangle(out, SELECTED_REGION_CORNERS[0], SELECTED_REGION_CORNERS[1],
                  panel["SELECTED_REGION_COLOR"], thickness=th*4, lineType=cv2.LINE_AA)

    # Define initial key info pixel coordinates
    key_info_p1 = [round(panel["KEY_INFO"][0][0]*fs), round(panel["KEY_INFO"][0][1]*fs)]
    key_info_p2 = [round(panel["KEY_INFO"][1][0]*fs), round(panel["KEY_INFO"][1][1]*fs)]

    # Draw legend box
    if panel["SHOW_KEY_INFO"]:
        ids_lists.append([])
        if len(ids_lists) > NUMBER_OF_FRAMES:
            if len(ids_lists[-NUMBER_OF_FRAMES]):
                cv2.rectangle(shapes, key_info_p1, [key_info_p2[0],
                              key_info_p2[1]*(len(ids_lists[-NUMBER_OF_FRAMES])+1)],
                              panel["KEY_INFO_COLOR"], cv2.FILLED)
            ids_lists.pop(0)

    # Generate output by blending image with shapes image, using the shapes
    # images also as mask to limit the blending to those parts
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(image, panel["KEY_INFO_TRANSPARENCY"], shapes,
                                1 - panel["KEY_INFO_TRANSPARENCY"], 0)[mask]

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

            # Calculate the predictions on the frame
            position, _, speed, future_move = calculate_predictions(reduced_box_data)

            if panel["PRINT_CALCULATIONS"]:
                print(f'\nid:{int(box_data[4])}.  Predicted position: {(position[0], position[1])}.'
                      f'  Predicted future_move: {future_move[0], future_move[1]}.')

            if (future_move != (0, 0) and not WAIT_FOR_BREAK_LINE):
                # Draw all dots
                number_of_inner_dots = round((np.linalg.norm(np.array(future_move)-
                                                             np.array((0, 0)))/
                                                            (panel["DOTS_SPACING"] * SCALING)))
                for factor in range(0, number_of_inner_dots + 2):
                    track_dot = (round(position[0]+factor/(number_of_inner_dots+1)*
                                        future_move[0]),
                                round(position[1]+factor/(number_of_inner_dots+1)*
                                        future_move[1]))
                    cv2.circle(out, track_dot, radius=3*image.shape[1]//1920,
                            color=panel["DOTS_COLOR"], thickness=-1)

                # Draw future object's box coordinates
                object_p1_after = (reduced_box_data[0] + future_move[0],
                            reduced_box_data[1] + future_move[1])
                object_p2_after = (reduced_box_data[2] + future_move[0],
                            reduced_box_data[3] + future_move[1])
                cv2.rectangle(out, object_p1_after, object_p2_after,
                            panel["FUTURE_BOX_COLOR"][int(box_data[-1])],
                            thickness=th, lineType=cv2.LINE_AA)

            # Draw info in legend box
            if (panel["SHOW_KEY_INFO"] and not WAIT_FOR_BREAK_LINE):
                ids_lists[-1].append(int(box_data[4]))
                if (len(ids_lists) == NUMBER_OF_FRAMES and
                    reduced_box_data[4] in ids_lists[0]):
                    ids_lists[0].sort()
                    speed_text = str(round(np.linalg.norm(np.array(speed)-
                                                            np.array((0, 0))), 2))
                    if speed_text == "0.0":
                        legend_text = f"id:{reduced_box_data[4]} is not moving."
                    else:
                        legend_text = f"id:{reduced_box_data[4]} moves at {speed_text} m/s."
                    cv2.putText(out, legend_text, (key_info_p1[0]*2, round(key_info_p2[1]*
                                (ids_lists[-NUMBER_OF_FRAMES].
                                index(reduced_box_data[4])+1.5))), fontFace=0, fontScale=fs,
                                color=panel["TEXTS_COLOR"], thickness=th,
                                lineType=cv2.LINE_AA)

            # Create trail of the current objects' position
            if panel["SHOW_TRAILS"]:
                trail = trails[reduced_box_data[4]]
                trail.append(position)
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

start_time = time.time()

# Loop through the video frames
while video.isOpened():
    start_time, _ = calculate_time_intervals(panel["PRINT_TIME_INTERVALS"], start_time,
                                             "cap.isOpened()")

    # Read a frame from the video
    ret, color_frame = video.read()
    start_time, _ = calculate_time_intervals(panel["PRINT_TIME_INTERVALS"], start_time,
                                             "read of image")

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
        start_time, _ = calculate_time_intervals(panel["PRINT_TIME_INTERVALS"], start_time,
                                                 "tracking")

        if panel["SHOW_TRACKING_PERFORMANCE"] or panel["SAVE_TRACKING_PERFORMANCE"]:
            # Store tracking performance
            preprocess_time.append(results[0].speed['preprocess'])
            inference_time.append(results[0].speed['inference'])
            postprocess_time.append(results[0].speed['postprocess'])

        # Plot the predictions on the frame
        output = plot_predictions(color_frame, results[0].boxes.data)
        start_time, _ = calculate_time_intervals(panel["PRINT_TIME_INTERVALS"], start_time,
                                                 "plot predictions")

        output = cv2.resize(output, tuple(panel["OUTPUT_RESOLUTION"]))

        if panel["SHOW_VIDEO"]:
            # Resize and display the annotated frame
            cv2.imshow('Chocolates tracking', output)
            start_time, _ = calculate_time_intervals(panel["PRINT_TIME_INTERVALS"], start_time,
                                                     "display of image")

        if panel["SAVE_VIDEO"]:
            frames.append(output)
            FRAME_COUNT = len(frames)
            print(f"\rFrame count : {FRAME_COUNT}    Seconds in video : "\
                  f"{round(FRAME_COUNT/CAMERA_FPS,2)}", end="")
            start_time, _ = calculate_time_intervals(panel["PRINT_TIME_INTERVALS"], start_time,
                                                     "save of image")

        key = cv2.waitKey(1)
        # Pause the loop if 'p' or 'P' or Space character is pressed
        if key in video_handling["pause"]:
            # wait until any key is pressed
            key = cv2.waitKey(-1)
        # Move forwards the loop if 'f' or 'F' is pressed
        if not panel["SAVE_VIDEO"] and key in video_handling["forward"]:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)+CAMERA_FPS
                                                   *panel["FUTURE_TIME"]))
            key = cv2.waitKey(1)
        # Move backwards the loop if 'b' or 'B' is pressed
        if not panel["SAVE_VIDEO"] and key in video_handling["backward"]:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)-CAMERA_FPS
                                                   *panel["PAST_TIME"]))
            key = cv2.waitKey(1)
        # Break the loop if 'q' or 'Q' or Esc character is pressed
        if key in video_handling["quit"]:
            print("\n\nVideo and tracking performance not saved.")
            break
    elif not panel["SAVE_VIDEO"]:
        # Repeat the loop if the end of the video is reached
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()

if panel["SAVE_VIDEO"] and key not in video_handling["quit"]:
    # Save the concatenated video as an MP4 file
    timestamp = time.strftime("%Y%m%d_%H%M%S_", time.localtime(time.time()))
    output_file = f"{timestamp}"+\
                  os.path.splitext(os.path.basename(__file__))[0] + ".mp4"

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


if panel["SHOW_TRACKING_PERFORMANCE"] or (panel["SAVE_TRACKING_PERFORMANCE"] and
                                          key not in video_handling["quit"]):

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

import os
import cv2
import logging as log
import time
from argparse import ArgumentParser
import numpy as np

from mouse_controller import MouseController
from input_feeder import InputFeeder
from face_detection import FaceDetector
from head_pose_estimation import HeadPoseEstimator
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimator

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-mfd", "--model_face_detection", required=True, type=str,
                        help="Path to an xml file with a trained face detection model.")
    parser.add_argument("-mhpe", "--model_head_pose_estimation", required=True, type=str,
                        help="Path to an xml file with a trained head pose estimation model.")
    parser.add_argument("-mfld", "--model_facial_landmarks_detection", required=True, type=str,
                        help="Path to an xml file with a trained facial landmarks detection model.")
    parser.add_argument("-mge", "--model_gaze_estimation", required=True, type=str,
                        help="Path to an xml file with a trained gaze estimation model.")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Specify 'video', 'image' or 'cam' (to work with camera).")
    parser.add_argument("-i", "--input_path", required=False, type=str, default=None,
                        help="Path to image or video file.")
    parser.add_argument("-o", "--output_path", required=False, type=str, default="results",
                        help="Path to image or video file.")                        
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    parser.add_argument("--show_input", help="Optional. Show input video",
                      default=False, action="store_true")
    parser.add_argument("--move_mouse", help="Optional. Move mouse based on gaze estimation",
                      default=False, action="store_true")
    return parser

def infer_on_stream(args):
    try:
        log.basicConfig(
            level=log.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                log.FileHandler("app.log"),
                log.StreamHandler()
            ])
            
        mouse_controller = MouseController(precision="low", speed="fast")

        start_model_load_time=time.time()

        face_detector = FaceDetector(args.model_face_detection)
        facial_landmarks_detector = FacialLandmarksDetector(args.model_facial_landmarks_detection)
        head_pose_estimator = HeadPoseEstimator(args.model_head_pose_estimation)
        gaze_estimator = GazeEstimator(args.model_gaze_estimation)
        face_detector.load_model()
        facial_landmarks_detector.load_model()
        head_pose_estimator.load_model()
        gaze_estimator.load_model()

        total_model_load_time = time.time() - start_model_load_time
        log.info("Model load time: {:.1f}ms".format(1000 * total_model_load_time))

        output_directory = os.path.join(args.output_path + '\\' + args.device)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        feed = InputFeeder(args.input_type, args.input_path)
        feed.load_data()
        out_video = feed.get_out_video(output_directory)

        frame_counter = 0
        start_inference_time=time.time()
        total_prepocess_time = 0

        
        while True:
            try:
                frame = next(feed.next_batch())
            except StopIteration:
                break
            frame_counter += 1

            face_boxes = face_detector.predict(frame)
            for face_box in face_boxes:
                face_image = get_crop_image(frame, face_box)
                eye_boxes, eye_centers = facial_landmarks_detector.predict(face_image)
                left_eye_image, right_eye_image = [get_crop_image(face_image, eye_box) for eye_box in eye_boxes]
                head_pose_angles = head_pose_estimator.predict(face_image)
                gaze_x, gaze_y = gaze_estimator.predict(right_eye_image, head_pose_angles, left_eye_image)
                draw_gaze_line(frame, face_box, eye_centers, gaze_x, gaze_y)
                if args.show_input:
                    cv2.imshow('im', frame)
                if args.move_mouse:
                    mouse_controller.move(gaze_x, gaze_y)
                total_prepocess_time += face_detector.preprocess_time + facial_landmarks_detector.preprocess_time + \
                    head_pose_estimator.preprocess_time + gaze_estimator.preprocess_time
                break

            if out_video is not None:
                out_video.write(frame)
            if args.input_type == "image":
                cv2.imwrite(os.path.join(output_directory, 'output_image.jpg'), frame)

            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                break
        
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=frame_counter/total_inference_time
        log.info("Inference time:{:.1f}ms".format(1000* total_inference_time))
        log.info("Input/output preprocess time:{:.1f}ms".format(1000* total_prepocess_time))
        log.info("FPS:{}".format(fps))

        with open(os.path.join(output_directory, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(total_prepocess_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')
            
        feed.close()
        cv2.destroyAllWindows()
    except Exception as e:
        log.exception("Something wrong when running inference:" + str(e))

def get_crop_image(image, box):
    xmin, ymin, xmax, ymax = box
    crop_image = image[ymin:ymax, xmin:xmax]
    return crop_image


def draw_gaze_line(image, face_box, eye_centers, gaze_x, gaze_y):
    xmin, ymin, xmax, ymax = face_box
    for x, y in eye_centers:
        start = (x+xmin, y+ymin)
        end = (x+xmin+int(gaze_x*3000), y+ymin-int(gaze_y*3000))
        beam_image = np.zeros(image.shape, np.uint8)
        for t in range(20)[::-2]:
            cv2.line(beam_image, start, end, (0, 0, 255-t*10), t*2)
        image |= beam_image

def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)

if __name__ == '__main__':
    main()
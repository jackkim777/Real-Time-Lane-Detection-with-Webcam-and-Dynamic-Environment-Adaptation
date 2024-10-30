수정된 README 파일은 다음과 같이 작성할 수 있습니다. 원래 내용을 유지하되, 수정한 점과 라이선스를 추가하였습니다.

```markdown
## Real-Time Lane Detection with Webcam and Dynamic Environment Adaptation
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* **Implement real-time lane detection using a webcam**.
* **Adapt the detection algorithm to varying weather and environmental conditions** for improved performance.

The images for camera calibration are stored in the folder called `camera_cal`. The images in `test_images` are for testing your pipeline on single frames.

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions. The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there! We encourage you to go out and take video of your own, calibrate your camera, and show us how you would implement this project from scratch!

## Usage:

### 1. Set up the environment 
```bash
conda env create -f environment.yml
```

To activate the environment:

- Windows: `conda activate carnd`
- Linux, MacOS: `source activate carnd`

### 2. Run the pipeline:
```bash
python lane_detection.py  # For real-time lane detection with webcam
python lane_detection.py --video INPUT_VIDEO OUTPUT_VIDEO_PATH  # For video input
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The original project was based on Udacity's Advanced Lane Lines project, which is available [here](https://github.com/udacity/CarND-Advanced-Lane-Lines).
```

이 수정된 README는 원래 프로젝트의 내용을 포함하면서, 당신이 추가한 기능과 라이선스 정보를 명확히 반영하고 있습니다. 필요한 경우 추가적인 내용을 더하거나 조정하셔도 좋습니다!
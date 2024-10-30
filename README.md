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
python main1.py  # For real-time lane detection with webcam
python main1.py --video INPUT_VIDEO OUTPUT_VIDEO_PATH  # For video input
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The original project was based on Udacity's Advanced Lane Lines project, which is available [here](https://github.com/udacity/CarND-Advanced-Lane-Lines).
```

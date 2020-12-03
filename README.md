![](logo/face_mask_detector_logo.svg)

## Introduction

The year 2020 is dominated by the global COVID-19 pandemic. People wearing face masks, to protect themselves against an infection, is an omnipresent image.



## Project Structure

![](docs/img/ps_full_dark.svg)



## General Usage

1. Clone the Git repository: `https://github.com/samuelrothen/face_mask_detector.git`
2. Install the requirements: `pip install requirements.txt`
3. Run `/src/live_video_detection.py` to start the live video detection



## Arduino Usage
The usage of an Arduino is disabled by default. If you want to use an Arduino to control the camera positioning follow the following steps:

1. Connect a servo motor to your Arduino using the following wiring:

![](docs/img/arduino_circuit.PNG)

2. Upload the `.ino`-Sketch from `src/aduino_sketch/python_servo_sketch/python_servo_sketch.ino` using the [Arduino IDE](https://www.arduino.cc/en/software)

3. Open `/src/live_video_detection.py` and set `use_arduino` to `True` (default is `False`)  and define your COM-Port in `serial.Serial(...)`

```python
use_arduino = False
if use_arduino:
    arduino = serial.Serial('COM3', 9600)
```



## License

Distributed under the MIT License. See `LICENSE` for more information.



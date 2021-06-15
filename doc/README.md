# Camera and Lidar Collision Detection
![Sensor-Fusion][1]
![Grad][2]
![Computer-Vision][3]
![Open-CV][4]

[1]: https://img.shields.io/:Sensor-Fusion-darkgreen.svg?style=round-square
[2]: https://img.shields.io/:Grad-Project-blue.svg?style=round-square
[3]: https://img.shields.io/:Computer-Vision-yellow.svg?style=round-square
[4]: https://img.shields.io/:Open-CV-purple.svg?style=round-square
[5]: https://img.shields.io/:Yolo-v3-p.svg?style=round-square
[6]: https://img.shields.io/:Covid-19-red.svg?style=round-square
---

## Overview
The project is modified to adapt with our vehicle system. GUI is suppressed and all we have to deal with is TTCs (Time-to-Collision), whether for Lidar or our mono-camera.

The project returns two parameters (camera TTC and lidar TTC), which are significant for detecting collision to just avoid it at the right time.

The project is just one publisher node in ROS. It is therefore communicate with the subscriber node using custom message (message passing).

---

## Assumptions and Notes
Since the process takes more or less 1.3 seconds, which is relatively long time, two measures are taken:
* We receive each frame, however, if the two successive frames are received while the current process is still working, we simply **drop them**.
* TTC are sent as `time = actual TTC - processing time`
* Frames are in the processing node. Yet, for further implmentations they shall be sent through a driver node.

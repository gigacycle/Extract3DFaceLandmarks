# Extract3DFaceLandmarks

This Python project extracts 3D landmarks based on 2D screenshots of a 3D face, including Lateral Left, Lateral Right, and Front View. It has undergone testing with the ESRC dataset curated by Peter Hancock at Stirling and Bernie Tiddeman at Aberystwyth. You can find his work [here](https://pics.stir.ac.uk/ESRC/index.htm).

![M1000_N](https://github.com/gigacycle/Extract3DFaceLandmarks/assets/2722068/c4671286-393b-4a74-91ad-720554ffb7b2)

Before utilizing this project, ensure that the facial object is aligned with the Frankfort Horizontal Plane. Detailed instructions on this alignment process can be found [here](https://www.otoscape.com/eponyms/frankfort-horizontal-plane.html). Note that all ESRC 3D Faces are already aligned with the Frankfort Horizontal Plane.

## Installation

Begin by cloning the project with the following command:

```bash
git clone https://github.com/gigacycle/Extract3DFaceLandmarks.git
```

Next, install the required dependencies using the following command:

```bash
pip install face-alignment, torchvision, vedo
```

## Usage

After successfully cloning the project and installing the dependencies, Run `main.py` and follow the instructions.

![VisualzeExtractedLandmarks](https://github.com/gigacycle/Extract3DFaceLandmarks/assets/2722068/73ac0941-3b76-404a-adb1-e481b7ce8d60)

1. Starts extracting 3D landmarks for ESRC dataset (stores at ./dataset/)
2. Starts extracting your own 3D face object (it needs full path of your .obj) and displays the landmarks at the end.
3. Displays the landmarks extracted in phase `1` (you should type the item name like the picture above)

## Acknowledgements

This project leverages the face-alignment and vedo libraries. 

- Face-alignment: [Link to the project](https://github.com/1adrianb/face-alignment)
- Vedo: [Link to the website](https://vedo.embl.es/)

Please ensure proper citation of these projects in your work.

Contributions to this project are welcomed via the submission of issues or pull requests.

For inquiries or assistance, kindly contact me.

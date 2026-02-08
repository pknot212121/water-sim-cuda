# MLS-MPM fluid simulation

## About
A fluid simulation with collisions using SDFs and great customisability. Both the initial fluid positions and collision objects are based on 3d .obj models. Also works mostly in real time (depends heavily on DT parameter).

## Setup Guide


## Controls

While inside the simulation you can use the following controls:

| Key    | Action                                                                  |
|--------|-------------------------------------------------------------------------|
| WASD   | Controls to move through the simulation on two axis                     |
| P      | Starts the simulation                                                   |
| L      | Changes display type of objects to wireframe (off by default)           |
| G      | Changes display type of objects to translucent (glass) (off by default) |
| F11    | Fullscreen                                                              | |

## Config

The config file functions on the basis of:
```
PARAMETER_NAME1=0.5
PARAMETER_NAME2=1000
PARAMETER_NAME3=file.txt
```
- All parameters must be placed in separate lines
- Spaces do not matter and are trimmed by the parser

### Mandatory parameters
If any one of these parameters is deleted or formatted incorrectly, the simulation will not start. Their types are as following:
```
SIZE_X=int
SIZE_Y=int
SIZE_Z=int

PADDING=int
GRAVITY=float
DT=float
GAMMA=float
COMPRESSION=float
RESOLUTION=float
SUBSTEPS=int
SDF_RESOLUTION=int
```
### Model parameters
With these you can load 3D models into the simulation. Only .obj files are supported currently. They need to be formatted as following:
```
WATER dir/model.obj;scale(float);moveX(float);moveY(float);moveZ(float)
OBJECT dir/model.obj;scale(float);moveX(float);moveY(float);moveZ(float)
```

# MLS-MPM fluid simulation

## About
A fluid simulation with collisions using SDFs and great customizability. Both the initial fluid positions and collision objects are based on 3d .obj models. Also works mostly in real time (depends heavily on DT parameter).

## Requirements
Should work on all computers with nvidia graphics cards with pascal architecture (10xx) or higher. Tested only on GTX 1050 and RTX 3060.


## Setup Guide

- Go into the releases page for [linux](https://github.com/pknot212121/water-sim-cuda/releases/tag/linux) or for windows.
- Download and unpack the zip file inside
- Activate the executable in the file

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
- It can decode integers, floats and strings

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
Where WATER means this model will be voxelized and filled with water particles and OBJECT means that the model will become a collision object.
Example:
```
WATER=models/sphere.obj;15.0;10.0;20.0;25.0
OBJECT=models/blender/u.obj;110.0;0.0;-10.0;0.0
```
- You can add as many models as you like, they can intersect and also go outside the grid.
- Scale parameter means the number of grid cells that the largest coordinate will be normalized to.
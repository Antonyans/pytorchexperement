# MaskDetection Feature

`__init__(path="somepath", device="cpu")`

**device**: {"cpu", "cuda"}

### Methods

|                  Methods                  |                                 Description                                   |
|:----------------------------------------: |:---------------------------------------------------------------------------:  |
| `__call__(self, facedet_response)`    | Gets response from face detection module and returns the response     |
|        `set_device(self, device)`         |                          Puts model onto the device                           |

---------------------------------

`__call__` _(self, facedet_response)_

**Parameters**

&nbsp;&nbsp;&nbsp;&nbsp;**facedet_response**: list, dict

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;either list of responses of face detection on `image` or one of the responses

**Return**

&nbsp;&nbsp;&nbsp;&nbsp;Returns a list of dict same length as `facedet_response`(order is preserved)

##### dict keys:

> **value**: str

> **probability** float [0,1]

#### Example
```
[{'probability': 0.52937690168619156, 'value': 'Masked'},
 {'probability': 0.807172879576683, 'value': 'Unmasked'}]
```

------------------

`set_device` _(self, device)_

**device** {"cpu", "cuda"}

loads the model into the `device`

<hr>

# DEPLOYMENT

## Weights file
Download [mask weights file](https://drive.google.com/file/d/18AosQQaVDpY2PLQxOAwMTs1uaa-8mzpL/view?usp=sharing) to `weights` folder.

## Environment variables
**DEVICE  
FACE_DETECTION_URL**


_The default value for DEVICE variable is `cpu`._  
_Please run container with `--gpus all` and set DEVICE=`cuda` to use the GPU._

Diverse motion data of a two-segment kinematic chain interconnected by a spherical joint.
Download the data as a .zip-archive from [link](https://faubox.rrze.uni-erlangen.de/getlink/fi6Z8QxcLV6B7V7HqiW5d6/data.zip) and unzip it.

# Data Descriptor

The resulting `data` folder contains 8192 sequences of 60s length at 100Hz. Each sequence contains a dictionary of arrays. Each array has shape `(6000)` and dtype `np.float32`. They dictionary contains:
- acc1_x, m/s**2
- acc1_y, 
- acc1_z, 
- acc2_x, 
- acc2_y,
- acc2_z, 
---
- gyr1_x, rad/s
- gyr1_y, 
- gyr1_z, 
- gyr2_x, 
- gyr2_y, 
- gyr2_z, 
---
- imu_to_joint1_x, m, the x-component of the (constant) vector pointing from the center of imu1 to the center of the spherical joint that connects the two segments
- imu_to_joint1_y,
- imu_to_joint1_z,
- imu_to_joint2_x, m, ... pointing from the center of imu2 to the ...
- imu_to_joint2_y,
- imu_to_joint2_z,
---
- quat1_u, unitless, the 0-th component of the quaternion that specifies the rotation at time t from imu1 to the earth frame, this is only provided for debugging, should not be used as a feature
- quat1_x,
- quat1_y,
- quat1_z,
---
- quat2_u, unitless, the 0-th components of the quaternion that speciifies the rotation at time t from imu2 to imu1, this rotation the *estimation target*
- quat2_x,
- quat2_y,
- quat2_z

# Target Metric
The goal is to accurately predict the relative rotation between imu1 and imu2, i.e., the `quat2`. The error between `quat2` and its predicition can be quantified by

```python
def angle_error(q, qhat):
    "Absolute angle in radians between `q` and `qhat`."
    return jnp.abs(quat_angle(quat_mul(quat_inv(q), qhat)))
```

which you can find at [link to GitHub Repo](https://github.com/simon-bachhuber/ring/blob/68e5e7739fdec68051ff0ea923f4f662021fde4f/src/ring/maths.py#L88)

# Convenience function
```python
import h5py

def load_array_dict(filename):
    """
    Loads a dictionary of NumPy arrays from an HDF5 file.

    Parameters:
    filename (str): Name of the HDF5 file to load the arrays from.

    Returns:
    dict: A dictionary where keys are dataset names and values are loaded NumPy arrays.
    """
    array_dict = {}
    with h5py.File(filename, "r") as h5file:
        for key in h5file.keys():
            array_dict[key] = h5file[key][:]
    return array_dict

load_array_dict("data/seq0.h5")
```
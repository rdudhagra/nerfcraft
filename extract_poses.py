from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation
import pyexiv2
import os
import utm
import json

from multiprocessing import Pool

# Variables
DATA_DIR = "./data/"
CAMERA_PARAMS = {
    "camera_angle_x": None,
    "camera_angle_y": None,
    "fl_x": 3.0869e03,
    "fl_y": 3.0794e03,
    "k1": 0.0180,
    "k2": -0.0737,
    "p1": 0,
    "p2": 0,
    "cx": 1.9906e03,
    "cy": 1.1116e03,
    "w": 4000.0,
    "h": 2250.0,
    "aabb_scale": 4,
}

# Some post-processing
CAMERA_PARAMS["camera_angle_x"] = 2 * np.arctan(
    CAMERA_PARAMS["w"] / (2 * CAMERA_PARAMS["fl_x"])
)
CAMERA_PARAMS["camera_angle_y"] = 2 * np.arctan(
    CAMERA_PARAMS["h"] / (2 * CAMERA_PARAMS["fl_y"])
)

# Helper functions
def dms_to_decimal(dms):
    """Converts a DMS string to a decimal number"""
    dms = dms.split(" ")
    d_frac = dms[0]
    m_frac = dms[1]
    s_frac = dms[2]

    d = float(d_frac.split("/")[0]) / float(d_frac.split("/")[1])
    m = float(m_frac.split("/")[0]) / float(m_frac.split("/")[1])
    s = float(s_frac.split("/")[0]) / float(s_frac.split("/")[1])

    return d + (m / 60) + (s / 3600)


def process_image(f):
    filepath = os.path.join(DATA_DIR, f)

    # Read the image
    img = pyexiv2.Image(filepath)
    xmp_data = img.read_xmp()
    exif_data = img.read_exif()
    img.close()

    # Parse the data
    lat_str = exif_data["Exif.GPSInfo.GPSLatitude"]
    lon_str = exif_data["Exif.GPSInfo.GPSLongitude"]

    # Convert the lat/lon from DMS to decimal
    lat = dms_to_decimal(lat_str)
    lon = dms_to_decimal(lon_str)

    # Convert the lat/lon to UTM (to get position in meters from some fixed origin)
    x, y, _, _ = utm.from_latlon(lat, lon)

    # Get altitude
    alt_m = float(xmp_data["Xmp.drone-dji.RelativeAltitude"])

    # Get gimbal pose
    gimbal_pitch_deg = float(xmp_data["Xmp.drone-dji.GimbalPitchDegree"])
    gimbal_roll_deg = float(xmp_data["Xmp.drone-dji.GimbalRollDegree"])
    gimbal_yaw_deg = float(xmp_data["Xmp.drone-dji.GimbalYawDegree"])
    # flight_pitch_deg = float(xmp_data["Xmp.drone-dji.FlightPitchDegree"])
    # flight_roll_deg = float(xmp_data["Xmp.drone-dji.FlightRollDegree"])
    flight_yaw_deg = float(xmp_data["Xmp.drone-dji.FlightYawDegree"])
    pitch_deg = gimbal_pitch_deg
    roll_deg = gimbal_roll_deg
    yaw_deg = -(gimbal_yaw_deg + flight_yaw_deg) - 90

    # Rotation matrix
    r = Rotation.from_euler(
        "zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True
    ).as_matrix()

    # Make the homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = np.array([x, y, alt_m])

    # Compute the sharpness
    im = Image.open(filepath).convert("L")  # to grayscale
    array = np.asarray(im, dtype=np.int32)
    im.close()

    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)

    # Construct the metadata frame
    new_frame = {}
    new_frame["file_path"] = filepath
    new_frame["sharpness"] = sharpness  # What does this do? I don't know
    new_frame["transform_matrix"] = T.tolist()

    return new_frame


if __name__ == "__main__":
    # Create metadata structure
    metadata = {}

    # Add the camera parameters
    metadata.update(CAMERA_PARAMS)

    metadata["frames"] = []

    # Find all the files
    files = [f for f in os.listdir(DATA_DIR) if ".JPG" in f]
    files.sort()

    # Initial processing of the files: extract metadata
    p = Pool()
    metadata["frames"] = p.map(process_image, files)

    # Post-processing: normalize poses to be centered at (0,0)
    avg_x = 0
    avg_y = 0
    for frame in metadata["frames"]:
        T = np.array(frame["transform_matrix"])
        avg_x += T[0, 3]
        avg_y += T[1, 3]

    avg_x /= len(metadata["frames"])
    avg_y /= len(metadata["frames"])

    for frame in metadata["frames"]:
        T = np.array(frame["transform_matrix"])
        T[0, 3] -= avg_x
        T[1, 3] -= avg_y
        frame["transform_matrix"] = T.tolist()

    # Save the metadata
    with open("transform.json", "w") as f:
        json.dump(metadata, f, indent=2)

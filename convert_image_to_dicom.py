"""
Converts a JPG/PNG image in a minimal DICOM (Secondary Capture)

Usage example
python3 convert_image_to_dicom.py -i xray.jpg -o xray.dcm --patient "DOE^John" --id 12345 --modality DX --monochrome
"""

import argparse
import datetime
import os

#pydicom is the library to create/manipulate DICOM datasets and files
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRBigEndian, SecondaryCaptureImageStorage

# Pillow handles reading common image formats (JPG/PNG)
from PIL import Image

# numpy is used to convert the Pillow image into raw bytes consistently
import numpy as np

def create_dicom_from_image(
        img_path,
        out_path,
        patient_name="ANON^PATIENT",
        patient_id="0000",
        modality="OT",
        study_uid=None,
        series_uid=None,
        sop_instance_uid=None,
        sop_class_uid=SecondaryCaptureImageStorage,
        make_monochrome=True
        ):
    """
    Create and save a DICOM file from an input image file

    Key design decisions:
    - Secondary Capture SOP class is used by default because it's generic and viewable by most DICOM viewers
    - Explicit VR little Edian Transfer syntax is chosen for the file dataset (common and compatible)
    - The File Meta (group 0002) is always written in Explicit VR Little Endian per DICOM standard
    - Pixel data is written uncompressed, if we where to use this for production we may need to use JPEG/JPEG2000 and change TransferSyntax accordingly
    """

    # 1. Read the input image with Pillow
    img = Image.open(img_path)
    #convert to MONOCHROME2 if requested (typical for x-ray)
    if make_monochrome:
        img = img.convert('L') # 'L' is a 8-bit grayscale
        photometric = "MONOCHROME2"
        samples_per_pixel = 1
    else:
        #convert to RGB (3 channels)
        img = img.convert("RGB")
        photometric = "RGB"
        samples_per_pixel = 3

    # Convert to numpy array for easy byte extraction and shape info
    arr = np.array(img)

    # 2. Convert numpy array for easy byte extraction and shape info
    file_meta = Dataset()
    # which SOP class (type of object) - using Secondary Capture by default
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    # SOP instance UID for this particular file
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid or generate_uid()
    # Transfer Syntax for file meta: use Explicit VR Little Endian for compatibility
    # (use the correct attribute name TransferSyntaxUID)
    from pydicom.uid import ExplicitVRLittleEndian
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    #implementationClassUID is optional but recommended (identify the software)
    file_meta.ImplementationClassUID = generate_uid()

    # 3. Create the FileDataset object -> this represents the whole DICOM file
    # the preamble (128 bytes) is optional but common; FileDataset handles "DICM" prefix after the preamble
    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b'\0' * 128)

    # 4. Populate required dataset (non-0002) attributes
    now = datetime.datetime.now()
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.Modality = modality # e.g., 'DX' for additional X-ray, 'OT' for other/secondary capture
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    # SOP Class & Instance UID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    # Study/Series/Instance dates and times
    ds.StudyDate = now.strftime('%Y%m%d')
    ds.StudyTime = now.strftime('%H%M%S')
    # Instance creation date/time (separate attributes)
    ds.InstanceCreationDate = now.strftime('%Y%m%d')
    ds.InstanceCreationTime = now.strftime('%H%M%S')

    #5. Image attributes tags - these must match the pixel buffer shape and type
    if samples_per_pixel == 1:
        # greyscale image - arr shape is (rows, cols)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = photometric # "MONOCHROME2"
        ds.Rows, ds.Columns = arr.shape
    else:
        # RGB image - arr shape is (rows, cols, channels)
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = photometric # 'RGB'
        ds.Rows, ds.Columns, _ = arr.shape

        # For RGB images, specify Planar Configuration (0 = chunky RGBRGB...)
        ds.PlanarConfiguration = 0

    # Bits per sample: we read 8-bit images here
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0 # 0 = unsigned integer

    #6. PixelData: raw bytes must match Rows*Columns*SamplePerPixel*(BitsAllocated/8)
    # for simple uncompressed images we can just pack the numpy array to bytes
    # ensure the data layout is contiguous and in the expected order
    if samples_per_pixel == 1:
        # for greyscale, arr.tobytes() yields row-major bytes which match the DICOM expection
        ds.PixelData = arr.tobytes()
    else:
        # for RGB, ensure the bytes order matches PlanarConfiguration
        # we used PlanarConfiguration = 0, so arr.tobytes() (row-major RGBRGB....) is fine
        ds.PixelData = arr.tobytes()

    #7. Transfer syntax and VR encoding for writing
    # file meta is Explicit VR Little Endian; set dataset flags accordingly
    ds.is_little_endian = True
    ds.is_implicit_VR = False # Explicit VR

    #8. Safe file - write_like_original=False ensures pydicom writes a proper file meta header
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    ds.save_as(out_path, write_like_original=False)
    print(f"Saved DICOM to {out_path}")
    return out_path

def main():
    p = argparse.ArgumentParser(description="Convert image to DICOM (Secondary Capture)")
    p.add_argument('-i', '--input', required=True, help='Input image (jpg/png)')
    p.add_argument('-o', '--output', required=True, help='Output DICOM filename (.dcm)')
    p.add_argument('--patient', default='ANON^PATIENT', help='PatientName (DICOM PN format)')
    p.add_argument('--id', default='0000', help='PatientID')
    p.add_argument('--modality', default='DX', help='Modality (e.g., DX, CR, OT)')
    p.add_argument('--monochrome', action='store_true', help='Convert image to MONOCHROME2 (recommended for X-ray)')
    args = p.parse_args()

    create_dicom_from_image(
        args.input,
        args.output,
        patient_name=args.patient,
        patient_id=args.id,
        modality=args.modality,
        make_monochrome=args.monochrome
    )


if __name__ == '__main__':
    main()

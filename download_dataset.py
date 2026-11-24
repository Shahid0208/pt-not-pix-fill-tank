"""
Download the oil tank satellite image dataset from Google Drive.

Usage:
    python download_dataset.py                    # downloads to ./data/
    python download_dataset.py -o /path/to/dir    # custom output directory
"""

import argparse
import gdown
import sys

GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1Mt5d0MJ55lsufTEDJ2pg3if-pWbiZhzz"


def main():
    parser = argparse.ArgumentParser(description="Download oil tank dataset from Google Drive")
    parser.add_argument("-o", "--output", default="data", help="Output directory (default: ./data)")
    args = parser.parse_args()

    print(f"Downloading to: {args.output}/")
    print(f"URL: {GOOGLE_DRIVE_URL}\n")

    try:
        output = gdown.download_folder(
            url=GOOGLE_DRIVE_URL,
            output=args.output,
            quiet=False,
        )
    except Exception as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not output:
        print("No files downloaded.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {len(output)} file(s) saved to {args.output}/")


if __name__ == "__main__":
    main()

"""
Full list of GDELT files: 
https://data.gdeltproject.org/gdeltv2/masterfilelist.txt
(Updated every 15 min)

Format follows:
    - 150383 297a16b493de7cf6ca809a7cc31d0b93 \
      http://data.gdeltproject.org/gdeltv2/20150218230000.export.CSV.zip

    - 318084 bb27f78ba45f69a17ea6ed7755e9f8ff \
      http://data.gdeltproject.org/gdeltv2/20150218230000.mentions.CSV.zip

    - 10768507 ea8dde0beb0ba98810a92db068c0ce99 \
      http://data.gdeltproject.org/gdeltv2/20150218230000.gkg.csv.zip

Note the streams:
    - 'export' = events
    - 'mentions' + 'gkg' = knowledge graph

Most recent three files: https://data.gdeltproject.org/gdeltv2/lastupdate.txt
"""

from dataclasses import dataclass
from datetime import date
import re


MASTER_FILE_LIST = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
LAST_UPDATE = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# Regex for filenames
FILENAME_REGEX = re.compile(
    r"(?P<timestamp>\d{14})\.(?P<stream>export|mentions|gkg)\.(?:CSV|csv)\.zip"
)


@dataclass
class GdeltFile:
    """
    Defines a GDELT file
    """

    url: str
    size_bytes: int
    md5: str
    stamp: str # YYYYMMDDHHMMSS
    stream: str # export | mentions | gkg

    # Date of file (YYYY-MM-DD)
    @property
    def day(self) -> date:
        return date(
            int(self.stamp[0:4]), 
            int(self.stamp[4:6]), 
            int(self.stamp[6:8])
        )

    # Full filename 
    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def parse_index(text: str) -> list[GdeltFile]:
    """
    Parse masterfilelist.txt or lastupdate.txt into GdeltFile records.

    Currently skips any malformed line (there is a lot more data to parse).
    """
    files: list[GdeltFile] = []

    for line in text.splitlines():
        parts = line.strip().split(" ")

        # Malformed
        if len(parts) != 3:
            continue

        size, md5, url = parts
        match = FILENAME_REGEX.search(url)

        # Malformed
        if not match or not size.isdigit():
            continue

        files.append(
            GdeltFile(
                url=url,
                size_bytes=int(size),
                md5=md5,
                stamp=match.group("stamp"),
                stream=match.group("stream")
            )
        )

    return files



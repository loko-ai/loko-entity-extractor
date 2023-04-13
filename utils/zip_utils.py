import os
import zipfile
from io import BytesIO
from pathlib import Path


def make_zipfile(output_path, files):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for fname, content in files:
            zip_file.writestr(fname, content.read())

def extract_zipfile(zip_file):
    files = []
    with zipfile.ZipFile(zip_file, 'r') as f:
        for fname in f.namelist():
            buffer = BytesIO()
            with f.open(fname) as source:
                buffer.write(source.read())
            buffer.seek(0)
            files.append([fname, buffer])
    return files



if __name__ == '__main__':
    path = Path('../resources')
    path = Path('/home/cecilia/Progetti/prove/')
    files = []
    for el in path.rglob('*'):
        buffer = BytesIO()
        if not el.is_dir():
            with open(el, 'rb') as f:
                buffer.write(f.read())
            buffer.seek(0)
            files.append([str(el.relative_to(path)), buffer])
        # else:
        #     files.append([str(el.relative_to(path)), ''])
    print(files)
    output_path = '/home/cecilia/Progetti/prove.zip'
    make_zipfile(output_path, files)
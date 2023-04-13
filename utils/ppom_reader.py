import json

def get_version():
    ppom=json.load(open("../ppom.json"))
    version=ppom['version']
    return version

def get_major_minor_version():
    version = get_version()
    return ".".join(version.split(".")[:2])

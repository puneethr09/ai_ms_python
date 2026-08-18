import argparse as arg
import pathlib as path

extension_list = {
        ".jpg": "Images",
        ".png": "Images",
        ".gif": "Images",
        ".jpeg": "Images",
        ".mp3": "Audio",
        ".wav": "Audio",
        ".mp4": "Video",
        ".avi": "Video",
        ".mkv": "Video",
        ".txt": "Text",
        ".pdf": "Documents",
        ".doc": "Documents",
        ".docx": "Documents",
        ".xls": "Documents",
        ".xlsx": "Documents",
        ".ppt": "Documents",
        ".pptx": "Documents",
        ".zip": "Archives",
        ".rar": "Archives",
        ".7z": "Archives",
        ".tar": "Archives",
        ".gz": "Archives",
        ".exe": "Others",
        ".msi": "Others",
        ".bat": "Others",
        ".sh": "Others",
        ".py": "Others",
        ".js": "Others",
        ".html": "Others",
        ".css": "Others",
        ".json": "Others",
        ".xml": "Others",
        ".csv": "Others",
        ".md": "Others",
        ".log": "Others",
        ".ini": "Others",
        ".cfg": "Others",
        ".conf": "Others",
        ".config": "Others",
        ".sys": "Others",
        ".dll": "Others",
        ".so": "Others",
        ".dylib": "Others",
        ".app": "Others",
        ".apk": "Others",
        ".ipa": "Others",
        ".dmg": "Others",
        ".iso": "Others",
        ".img": "Others",
        ".vhd": "Others",
        ".vhdx": "Others",
        ".vmdk": "Others",
        ".ova": "Others",
        ".ovf": "Others",
        ".qcow2": "Others",
        ".raw": "Others",
        ".bin": "Others",
        ".hex": "Others",
        ".rom": "Others",
    }

def arg_parse():
    '''
    parses the arguments
    '''
    parser = arg.ArgumentParser()
    parser.add_argument("--source", required=True, type=path.Path)
    parser.add_argument("--dest", required=True, type=path.Path)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()

def scan_directory(directory):
    '''
    scans the directory and returns a list of files
    '''
    files = directory.iterdir()
    return files

def interpret_extensions(filename):
    extension = filename.suffix.lower()
    return extension_list.get(extension, "Others")

def move_files(files, dest, category, dry):
    category_path = dest / category
    if dry:
        print("Dry run: ", files.name, "to", category_path)
        return
    else:
        category_path.mkdir(exist_ok=True)
        destination = category_path / files.name
        if destination.exists():
            print("File already exists: ", destination)
            return
        files.rename(destination)
        print("Moved: ", files.name, "to", destination)
    
def main():
    args = arg_parse()
    source = args.source
    dest = args.dest
    dry_run = args.dry_run
    files = scan_directory(source)
    for file in files:
        if file.is_file():
            category = interpret_extensions(file)
            move_files(file, dest, category,dry_run) 
        else:
            print("Skipping directory: ", file)


if __name__ == "__main__":
    main()
    
    
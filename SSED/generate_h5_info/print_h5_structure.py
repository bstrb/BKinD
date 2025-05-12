import h5py

def print_h5_structure_details(file_path):
    try:
        with h5py.File(file_path, 'r') as h5file:
            def print_attrs(name, obj):
                print("\nGeneral Object Info")
                print(f"Name: {name.split('/')[-1]}")
                print(f"Path: /{name}")
                print(f"Type: {'HDF5 Dataset' if isinstance(obj, h5py.Dataset) else 'Group'}")
                print(f"Object Ref: {obj.ref}")
                if isinstance(obj, h5py.Dataset):
                    print("Dataset Dataspace and Datatype")
                    for attr_name in dir(obj):
                        if not attr_name.startswith('_') and not callable(getattr(obj, attr_name)):
                            attr_value = getattr(obj, attr_name)
                            print(f"{attr_name}: {attr_value}")

            h5file.visititems(print_attrs)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

if __name__ == "__main__":
    # Example usage
    file_path = "/Users/xiaodong/mask/pxmask.h5"
    print_h5_structure_details(file_path)

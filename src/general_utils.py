from pygrabber.dshow_graph import FilterGraph


def select_camera():
    graph = FilterGraph()
    cameras = graph.get_input_devices()
    if not cameras:
        print("No cameras found.")
        return None
    print("Available cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i}: {cam}")
    cam_index = int(input("Select camera Number: "))
    return cam_index, cameras[cam_index]

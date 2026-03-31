import cv2

def test_resolution(cap, width, height, fourcc=None):
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    
    # Convert fourcc back to string
    v = int(actual_fourcc)
    f = chr(v & 0xFF) + chr((v >> 8) & 0xFF) + chr((v >> 16) & 0xFF) + chr((v >> 24) & 0xFF)
    
    print(f"Requested: {width}x{height} {fourcc if fourcc else ''}")
    print(f"Actual:    {int(actual_w)}x{int(actual_h)} {f}")
    
    return int(actual_w) == width and int(actual_h) == height

def main():
    cam_index = 0
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"Cannot open camera at index {cam_index}")
        return

    resolutions = [
        (640, 480),
        (720, 480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160)
    ]
    
    formats = ["MJPG", "YUYV"]

    for fmt in formats:
        print(f"\n--- Testing Format: {fmt} ---")
        for w, h in resolutions:
            success = test_resolution(cap, w, h, fmt)
            if success:
                print(f"  [SUCCESS] {w}x{h} supported with {fmt}")
            else:
                print(f"  [FAILED] {w}x{h} NOT supported with {fmt}")

    cap.release()

if __name__ == "__main__":
    main()

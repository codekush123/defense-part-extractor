import cv2
import os

def pick_coordinate(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return

    print("--- INSTRUCTIONS ---")
    print("1. A window will open with your image.")
    print("2. CLICK on the part you want to extract.")
    print("3. The X and Y coordinates will print in this terminal.")
    print("4. Press any key to close the window.")

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\n✅ Target Found!")
            print(f"X Coordinate: {x}")
            print(f"Y Coordinate: {y}")
            # Draw a small circle where you clicked to confirm
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Picker", img)

    cv2.namedWindow("Picker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Picker", click_event)
    cv2.imshow("Picker", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change '65.png' to whatever your file is named in data/raw
    target_image = os.path.join("data", "raw", "65.png")
    pick_coordinate(target_image)
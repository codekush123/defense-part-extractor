import cv2
import os

def pick_coordinate(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return

    points = []
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"Point {len(points)} added: {x}, {y}")
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Picker", img)
            
            if len(points) == 4:
                coord_str = ",".join([f"{p[0]},{p[1]}" for p in points])
                print(f"\nAll 4 points collected!")
                print(f"Use this string for your API: {coord_str}")
    cv2.namedWindow("Picker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Picker", click_event)
    cv2.imshow("Picker", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change '65.png' to whatever your file is named in data/raw
    target_image = os.path.join("data", "raw", "65.png")
    pick_coordinate(target_image)
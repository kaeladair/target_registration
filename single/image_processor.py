from typing import List, Tuple
import numpy as np
import cv2
import json

# Placeholder for the latlon2pixel function
def latlon2pixel(lat0, lon0, lat, lon, x, y, yaw, H_resolution, V_resolution):
    # Add your implementation of converting lat/lon to pixel coordinates here
    # This is a dummy implementation for illustration purposes
    return lat0 + (y / H_resolution), lon0 + (x / V_resolution)

class ImageProcessor:
    def __init__(self, image_path: str, latitude: float, longitude: float, yaw: int, area_threshold: float):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.latitude = latitude
        self.longitude = longitude
        self.yaw = yaw
        self.area_threshold = area_threshold
        self.contours = self.detect_contours()

    def detect_contours(self) -> List[np.ndarray]:
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.area_threshold]
        return sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    def get_main_contour_mask(self) -> np.ndarray:
        mask = np.zeros_like(self.image)
        if self.contours:
            cv2.drawContours(mask, [self.contours[0]], -1, (255), thickness=cv2.FILLED)
        return mask

    def geolocate_contours(self, H_resolution: float, V_resolution: float) -> List[Tuple[float, float]]:
        geolocations = []
        for contour in self.contours:
            for point in contour:
                lat, lon = latlon2pixel(self.latitude, self.longitude, 0, 0, point[0][0], point[0][1], self.yaw, H_resolution, V_resolution)
                geolocations.append((lat, lon))
        return geolocations

    def calculate_homography(self, other_image: 'ImageProcessor') -> np.ndarray:
        kp1a, kp1b, gm = find_goodmatches(self.image, other_image.image, self.get_main_contour_mask(), other_image.get_main_contour_mask(), 0.75)
        src_pts = np.float32([kp1a[m.queryIdx].pt for m in gm]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1b[m.trainIdx].pt for m in gm]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def apply_homography(self, H: np.ndarray, points: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(points, H)

def load_metadata(metadata_path: str):
    with open(metadata_path, 'r') as file:
        data = json.load(file)
    return data['images']

def find_goodmatches(im1a, im1b, mask1, mask2, RATIO):
    sift = cv2.SIFT_create()
    kp1a, des1a = sift.detectAndCompute(im1a, mask1)
    kp1b, des1b = sift.detectAndCompute(im1b, mask2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1a, des1b, k=2)
    good_matches = [m for m, n in matches if m.distance < RATIO * n.distance]
    return kp1a, kp1b, good_matches

def main():
    metadata = load_metadata("metadata.json")
    area_threshold = 500  # Adjust this threshold based on your needs
    images = [ImageProcessor(img_data['path'], img_data['latitude'], img_data['longitude'], img_data['yaw'], area_threshold) for img_data in metadata]

    global_canvas = np.zeros((10000, 10000, 3), dtype=np.uint8)
    all_transformed_contours = []

    for i in range(len(images) - 1):
        for j in range(i + 1, len(images)):
            H = images[i].calculate_homography(images[j])
            for contour in images[i].contours:
                contour = np.array(contour, dtype=np.float32).reshape(-1, 1, 2)
                transformed_contour = images[i].apply_homography(H, contour)
                all_transformed_contours.append(transformed_contour)
                
                # Display the original contour
                original_image = cv2.cvtColor(images[i].image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(original_image, [np.int32(contour)], -1, (0, 255, 0), 3)
                cv2.imshow("Original Contour", original_image)
                cv2.waitKey(0)

                # Display the transformed contour
                transformed_image = cv2.cvtColor(images[j].image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(transformed_image, [np.int32(transformed_contour)], -1, (0, 255, 0), 3)
                cv2.imshow("Transformed Contour", transformed_image)
                cv2.waitKey(0)

    if all_transformed_contours:
        for contour in all_transformed_contours:
            cv2.drawContours(global_canvas, [np.int32(contour)], -1, (0, 255, 0), 3)
    else:
        print("No transformed contours found.")

    cv2.imshow("Global Canvas with Hotspots", global_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


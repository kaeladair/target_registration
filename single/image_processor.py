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
    def __init__(self, image_path: str, latitude: float, longitude: float, yaw: int):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.latitude = latitude
        self.longitude = longitude
        self.yaw = yaw
        self.contours = self.detect_contours()

    def detect_contours(self) -> List[np.ndarray]:
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=cv2.contourArea, reverse=True)
    
    def get_main_contour_mask(self) -> np.ndarray:
        mask = np.zeros_like(self.image)
        cv2.drawContours(mask, [self.contours[0]], -1, (255), thickness=cv2.FILLED)
        return mask

    def geolocate_contours(self, H_resolution: float, V_resolution: float) -> List[Tuple[float, float]]:
        geolocations = []
        for contour in self.contours:
            for point in contour:
                lat, lon = latlon2pixel(self.latitude, self.longitude, 0, 0, point[0][0], point[0][1], self.yaw, H_resolution, V_resolution)
                geolocations.append((lat, lon))
        return geolocations

    def visualize_transformation(self, transformed_contours: List[np.ndarray], original_contours: List[np.ndarray]):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        canvas = np.zeros((max(self.image.shape[0], transformed_contours[0].shape[0]), 
                           self.image.shape[1] * 2, 3), dtype=np.uint8)
        
        for i, contour in enumerate(transformed_contours):
            color = colors[i % len(colors)]
            cv2.drawContours(canvas[:, :self.image.shape[1]], [np.int32(contour)], -1, color, 3)

        for i, contour in enumerate(original_contours):
            color = colors[i % len(colors)]
            cv2.drawContours(canvas[:, self.image.shape[1]:], [np.int32(contour)], -1, color, 3)
        
        cv2.imshow("Transformed and Original Contours", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    '''
    This function uses SIFT to find keypoints in two images,
    then it uses a BFmatcher with Knn=2 to select the best matches
    NOTE: im1a and im1b are in grayscale!
    '''
    # Apply SIFT and find correspondences
    sift = cv2.SIFT_create()

    # SIFT with masks
    kp1a, des1a = sift.detectAndCompute(im1a, mask1)
    kp1b, des1b = sift.detectAndCompute(im1b, mask2)

    # find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1a, des1b, k=2)
    print('Number of matches found:', len(matches))

    # Filter out potentially wrong outliers based on distance metric
    good_matches = []
    gm = []

    for m, n in matches:
        if m.distance < RATIO * n.distance:
            good_matches.append([m])
            gm.append(m)

    # Display matches over original (BGR) images
    print('Total number of filtered matches: ', len(gm))

    matched = cv2.drawMatchesKnn(im1a, kp1a, im1b, kp1b, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matched Keypoints", matched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return kp1a, kp1b, gm

def main():
    metadata = load_metadata("metadata.json")
    image1_data = metadata[0]
    image2_data = metadata[1]

    image1 = ImageProcessor(image1_data['path'], image1_data['latitude'], image1_data['longitude'], image1_data['yaw'])
    image2 = ImageProcessor(image2_data['path'], image2_data['latitude'], image2_data['longitude'], image2_data['yaw'])

    H = image1.calculate_homography(image2)
    
    # Apply homography to all contours
    transformed_contours = [image1.apply_homography(H, np.array(contour, dtype=np.float32).reshape(-1, 1, 2)) for contour in image1.contours]

    # Visualize both original and transformed contours
    image1.visualize_transformation(transformed_contours, image1.contours)

    # Geolocate all contours
    geolocations = image1.geolocate_contours(0.08612232500739336, 0.08333333333333333)
    print("Geolocated Hotspots:", geolocations)

if __name__ == "__main__":
    main()

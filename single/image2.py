import cv2
import numpy as np
from typing import List, Tuple

def apply_gaussian_blur(image, kernel_size=9):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_contours(image, area_threshold=1000, intensity_threshold=100) -> List[np.ndarray]:
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to reduce noise and close gaps
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and intensity
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mean_intensity = cv2.mean(image, mask=mask)[0]
            if mean_intensity > intensity_threshold:
                filtered_contours.append(contour)
            else:
                print(f"Contour rejected by intensity: Area = {area}, Intensity = {mean_intensity}")
        else:
            print(f"Contour rejected by area: Area = {area}")
    
    # Sort contours by area in descending order
    return sorted(filtered_contours, key=cv2.contourArea, reverse=True)

def draw_contours(image, contours, title="Contours"):
    img_contours = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)  # Ensure the image is in BGR format
    img_contours = cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)  # Drawing in red with thickness 2
    cv2.imshow(title, img_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_keypoints_and_descriptors(image, contour) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, mask)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2, ratio=0.75) -> List[cv2.DMatch]:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # Find the two nearest matches for each descriptor
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]  # Apply the ratio test
    return good_matches

def filter_matches_ransac(keypoints1, keypoints2, matches, reproj_threshold=3.0):
    if len(matches) < 4:
        return matches  # Not enough matches to compute homography

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)
    if H is None:
        return []  # No valid homography found

    inliers = [matches[i] for i in range(len(matches)) if mask[i]]
    return inliers

def main(image1_path, image2_path, area_threshold=1000, intensity_threshold=100, kernel_size=9):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Check if the images were loaded successfully
    if img1 is None:
        print(f"Error loading image1 from path: {image1_path}")
        return
    if img2 is None:
        print(f"Error loading image2 from path: {image2_path}")
        return

    # Apply Gaussian blur to both images
    img1_blur = apply_gaussian_blur(img1, kernel_size)
    img2_blur = apply_gaussian_blur(img2, kernel_size)

    # Detect contours with additional intensity filtering
    contours1 = detect_contours(img1_blur, area_threshold, intensity_threshold)
    contours2 = detect_contours(img2_blur, area_threshold, intensity_threshold)

    # Visualize original contours
    draw_contours(img1, contours1, title="Contours of Image 1")
    draw_contours(img2, contours2, title="Contours of Image 2")

    for contour1 in contours1:
        keypoints1, descriptors1 = extract_keypoints_and_descriptors(img1, contour1)
        
        for contour2 in contours2:
            keypoints2, descriptors2 = extract_keypoints_and_descriptors(img2, contour2)
            
            if descriptors1 is not None and descriptors2 is not None:
                matches = match_keypoints(descriptors1, descriptors2)
                inlier_matches = filter_matches_ransac(keypoints1, keypoints2, matches)

                # Draw matches
                img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow("Matches", img_matches)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    image1_path = 'images/case1a.png'
    image2_path = 'images/case1b.png'
    main(image1_path, image2_path)

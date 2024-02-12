import cv2
import numpy as np

def stitch_images(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Keep good matches: Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.2 * n.distance:
            good_matches.append(m)

    # Homography if enough matches are found
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        # Get the size of both images
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        # Get the canvas size by warping the corners of the images and finding the extremities
        corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners_img2 = cv2.perspectiveTransform(corners_img2, M)
        all_corners = np.concatenate((corners_img1, warped_corners_img2), axis=0)

        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

        result = cv2.warpPerspective(img1, Ht.dot(M), (xmax-xmin, ymax-ymin))
        result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2

        return result
    else:
        return None

# Load images
img1 = cv2.imread('./images/DSC02930.JPG')  # Replace with your image path
img2 = cv2.imread('./images/DSC02931.JPG')  # Replace with your image path

# Stitch images
result = stitch_images(img1, img2)

# Save or show result
if result is not None:
    cv2.imshow("Stitched Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("stitched.jpg", result)
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), 10))

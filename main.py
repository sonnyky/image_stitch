import cv2
import numpy as np

# Load images
image1 = cv2.imread('./images/DSC02930.JPG')
image2 = cv2.imread('./images/DSC02931.JPG')

# Initialize SIFT
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Feature matching
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Find homography
src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# Warp image2 to image1's perspective
warped_image2 = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))

# Create an empty canvas for the final stitched image
stitched_image = np.zeros((max(image1.shape[0], warped_image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)

# Place the warped image onto the canvas
stitched_image[0:warped_image2.shape[0], 0:warped_image2.shape[1]] = warped_image2

# Overlay image1 onto the canvas
# Note: You may need to adjust the range depending on the actual overlap
stitched_image[0:image1.shape[0], 0:image1.shape[1]] = image1

# Optional: Crop out the black areas or adjust the canvas size

# Desired width
new_width = 1920

# Calculate the new height maintaining the aspect ratio
aspect_ratio = stitched_image.shape[0] / stitched_image.shape[1]  # height/width ratio of the original image
new_height = int(new_width * aspect_ratio)

# Resize the image
resized_image = cv2.resize(stitched_image, (new_width, new_height))


# Show or save the result
cv2.imshow('Stitched Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
